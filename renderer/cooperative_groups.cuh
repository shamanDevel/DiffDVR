/**
 * Shortened version of cooperative_groups.h from the CUDA SDK
 */

#ifndef _CUSTOM_COOPERATIVE_GROUPS_H_
#define _CUSTOM_COOPERATIVE_GROUPS_H_

#if defined(__cplusplus) && defined(__CUDACC__)
namespace kernel
{
#if !defined(_CG_QUALIFIER)
# define _CG_QUALIFIER __forceinline__ __device__
# define _CG_STATIC_CONST_DECL static constexpr
#endif

    static _CG_QUALIFIER unsigned int lanemask32_lt()
    {
        unsigned int lanemask32_lt;
        asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask32_lt));
        return (lanemask32_lt);
    }
    static _CG_QUALIFIER unsigned int laneid()
    {
        unsigned int laneid;
        asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
        return laneid;
    }
	
    class coalesced_group;
    _CG_QUALIFIER coalesced_group coalesced_threads();
	
    class coalesced_group
    {
    protected:
        struct tg_data {
            unsigned int is_tiled : 1;
            unsigned int type : 7;
            unsigned int size : 24;
            // packed to 4b
            unsigned int metaGroupSize : 16;
            unsigned int metaGroupRank : 16;
            // packed to 8b
            unsigned int mask;
            // packed to 12b
            unsigned int _res;
        };
        tg_data _data;
    	
    private:
        friend _CG_QUALIFIER coalesced_group coalesced_threads();

        _CG_QUALIFIER unsigned int _packLanes(unsigned laneMask) const {
            unsigned int member_pack = 0;
            unsigned int member_rank = 0;
            for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
                unsigned int lane_bit = _data.mask & (1 << bit_idx);
                if (lane_bit) {
                    if (laneMask & lane_bit)
                        member_pack |= 1 << member_rank;
                    member_rank++;
                }
            }
            return (member_pack);
        }

    public:
        _CG_QUALIFIER coalesced_group(unsigned int mask) {
            _data.mask = mask;
            _data.size = __popc(mask);
            _data.metaGroupRank = 0;
            _data.metaGroupSize = 1;
            _data.is_tiled = false;
        }

        _CG_QUALIFIER unsigned int get_mask() const {
            return (_data.mask);
        }

        _CG_QUALIFIER unsigned int size() const {
            return (_data.size);
        }
    	
        _CG_QUALIFIER unsigned int thread_rank() const {
            return (__popc(_data.mask & lanemask32_lt()));
        }

        // Rank of this group in the upper level of the hierarchy
        _CG_QUALIFIER unsigned int meta_group_rank() const {
            return _data.metaGroupRank;
        }

        // Total num partitions created out of all CTAs when the group was created
        _CG_QUALIFIER unsigned int meta_group_size() const {
            return _data.metaGroupSize;
        }

        _CG_QUALIFIER void sync() const {
            __syncwarp(_data.mask);
        }

        template <typename TyIntegral>
        _CG_QUALIFIER TyIntegral shfl(TyIntegral var, unsigned int src_rank) const {
            unsigned int lane = (src_rank == 0) ? __ffs(_data.mask) - 1 :
                (size() == 32) ? src_rank : __fns(_data.mask, 0, (src_rank + 1));
            return (__shfl_sync(_data.mask, var, lane, 32));
        }

        template <typename TyIntegral>
        _CG_QUALIFIER TyIntegral shfl_up(TyIntegral var, int delta) const {
            if (size() == 32) {
                return (__shfl_up_sync(0xFFFFFFFF, var, delta, 32));
            }
            unsigned lane = __fns(_data.mask, laneid(), -(delta + 1));
            if (lane >= 32) lane = laneid();
            return (__shfl_sync(_data.mask, var, lane, 32));
        }

        template <typename TyIntegral>
        _CG_QUALIFIER TyIntegral shfl_down(TyIntegral var, int delta) const {
            if (size() == 32) {
                return (__shfl_down_sync(0xFFFFFFFF, var, delta, 32));
            }
            unsigned int lane = __fns(_data.mask, laneid(), delta + 1);
            if (lane >= 32) lane = laneid();
            return (__shfl_sync(_data.mask, var, lane, 32));
        }

        _CG_QUALIFIER int any(int predicate) const {
            return (__ballot_sync(_data.mask, predicate) != 0);
        }
        _CG_QUALIFIER int all(int predicate) const {
            return (__ballot_sync(_data.mask, predicate) == _data.mask);
        }
        _CG_QUALIFIER unsigned int ballot(int predicate) const {
            if (size() == 32) {
                return (__ballot_sync(0xFFFFFFFF, predicate));
            }
            unsigned int lane_ballot = __ballot_sync(_data.mask, predicate);
            return (_packLanes(lane_ballot));
        }

};
	
_CG_QUALIFIER coalesced_group coalesced_threads()
{
	return (coalesced_group(__activemask()));
}

template <typename TyVal, typename TyOp>
_CG_QUALIFIER auto shuffle_reduce_pow2(const coalesced_group& group, TyVal val, TyOp op) -> decltype(op(val, val)) {
    using TyRet = decltype(op(val, val));
    TyRet out = val;

    for (int offset = group.size() >> 1; offset > 0; offset >>= 1)
        out = op(out, group.shfl_down(out, offset));

    return out;
}
	
template <typename TyVal, typename TyOp>
_CG_QUALIFIER auto reduce(const coalesced_group& group, TyVal val, TyOp op) -> decltype(op(val, val)) {
    using TyRet = decltype(op(val, val));
    const unsigned int groupSize = group.size();
    bool isPow2 = (groupSize & (groupSize - 1)) == 0;
    TyRet out = val;

    // Normal shfl_down reduction if the group is a power of 2
    if (isPow2) {
        // Dispatch correct answer from lane 0 after performing the reduction
        return group.shfl(shuffle_reduce_pow2(group, val, op), 0);
    }
    else {
        const unsigned int mask = group.get_mask();
        unsigned int lanemask = lanemask32_lt() & mask;
        unsigned int srcLane = laneid();

        const unsigned int base = __ffs(mask) - 1; /* lane with rank == 0 */
        const unsigned int rank = __popc(lanemask);

        for (unsigned int i = 1, j = 1; i < groupSize; i <<= 1) {
            if (i <= rank) {
                srcLane -= j;
                j = i; /* maximum possible lane */

                unsigned int begLane = base + rank - i; /* minimum possible lane */

                /*  Next source lane is in the range [ begLane .. srcLane ]
                    *  If begLane < srcLane then do a binary search.
                    */
                while (begLane < srcLane) {
                    const unsigned int halfLane = (begLane + srcLane) >> 1;
                    const unsigned int halfMask = lanemask >> halfLane;
                    const unsigned int d = __popc(halfMask);
                    if (d < i) {
                        srcLane = halfLane - 1; /* halfLane too large */
                    }
                    else if ((i < d) || !(halfMask & 0x01)) {
                        begLane = halfLane + 1; /* halfLane too small */
                    }
                    else {
                        begLane = srcLane = halfLane; /* happen to hit */
                    }
                }
            }

            TyVal tmp = __shfl_sync(out, mask, srcLane, 32);
            if (i <= rank) {
                out = op(out, tmp);
            }
        }
        // Redistribute the value after performing all the reductions
        return group.shfl(out, groupSize - 1);
    }
}

template <typename Ty> struct plus
{
	__device__ __forceinline__ Ty operator()(Ty arg1, Ty arg2) const { return arg1 + arg2; }
};
	
}
#endif

#endif
