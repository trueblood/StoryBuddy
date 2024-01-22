import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, "/var/www/api.storybuddy.angrybuddy.com/")

from app import app as application

