import os
import copy
import yaml
import logging
from collections import namedtuple
from config.serializers import ConfigSchema

logger = logging.getLogger("Config")
logger.setLevel(logging.ERROR)

class Config(object):
    _cfg = None
    _errors = None

    def __init__(self, filename=None):
        if not filename:
            self.filename = "%s/config.yml" % os.getcwd()
        else:
            self.filename = "%s/%s" % (os.getcwd(), filename)

    def load(self):
        config_schema = ConfigSchema()
        try:
            with open(self.filename, 'r') as ymlfile:
                data = yaml.load(ymlfile)
            self._cfg, self._errors = config_schema.load(data)
            if self._errors:
                raise ValueError("Error in config files %s " % self._errors)
        except Exception as e:
            logger.error(e)
            raise ValueError("Can't open config file %s" % self.filename)

    @property
    def cfg(self):
        return self._cfg

    @property
    def cfgdata(self):
        attrs = []
        if not self._cfg:
            self.load()
        cfg = copy.deepcopy(self._cfg)
        for key in cfg:
            elm = cfg[key]
            attrs.append(key)
            if isinstance(elm, dict):
                namedTuple = namedtuple(key, sorted(elm.keys()))
                cfg[key] = namedTuple(**elm)
        config_tmp = namedtuple('config', attrs)
        if self._cfg:
            return config_tmp(**cfg)
        else:
            return None

