from datetime import datetime
import pytz
import os


class PrintFile():

  """Prints strings to a local file."""

  def __init__(self, parent_folder, *,
               filename = 'log.txt',
               timezone = 'UTC',
               ):
    """    
    Args:
        parent_folder (str): Path containing subfolder with date and time of the created file.
        filename (str, default "log.txt"): File name that will contain the strings
        timezone (str, default "UTC"): Label of the timezone as argumento for pytz.timezone function.
    """
    self.timezone = pytz.timezone(timezone)
    self.parent_folder = parent_folder
    self.filename = filename
    self.folder = None


  def _create_folder(self):
    "Create folder and file only if this is the first call"
    if self.folder is None:
      now = datetime.now(self.timezone)
      now = str(now).replace(' ', '_')
      self.folder = self.parent_folder + '/' + now + '/'
      os.system('mkdir -p ' + self.folder)
      self.filepath = self.folder + self.filename


  def __call__(self, text):
    """Adds the text content to a local file.
    
    Args:
        text (str): content to be added to a local file.
    """
    self._create_folder()
    print(text, file=open(self.filepath, 'a'))