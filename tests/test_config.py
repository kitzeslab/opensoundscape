#!/usr/bin/env python3
import pytest
import opensoundscape.config as config


def test_default_config_validates():
    assert config.validate(config.DEFAULT_CONFIG)
