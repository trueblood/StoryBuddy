import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, "/var/www/storybuddy.angrybuddy.com/")

from main import app as application

