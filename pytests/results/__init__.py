import os

def get_result_folder(name : str):
  """
  Returns the absolute path to a subfolder denoted by 'name' without a trailing slash.
  :param name: the name of the subfolder
  :return: the absolute path to that subfolder
  """
  current_dir = os.path.split(__file__)[0]
  path = os.path.abspath(os.path.join(current_dir, name))
  if os.path.exists(path):
    assert os.path.isdir(path)
  else:
    os.makedirs(path)
  return path