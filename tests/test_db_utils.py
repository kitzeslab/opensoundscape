#!/usr/bin/env python3
import sys

sys.path.append("..")
from modules.utils import generate_config
from modules.db_utils import write_ini_section, OpensoundscapeAttemptOverrideINISection
import pymongo
import pytest


@pytest.fixture()
def ini_test():
    return generate_config("../config/opensoundscape.ini", "ini_test.ini")


@pytest.fixture()
def ini_test_change():
    return generate_config("../config/opensoundscape.ini", "ini_test_change.ini")


@pytest.fixture()
def db(ini_test):
    with pymongo.MongoClient(ini_test["general"]["db_uri"]) as client:
        db = client[ini_test["general"]["db_name"]]
        return db


@pytest.fixture()
def coll(request, db):
    coll = db["ini"]

    def drop():
        coll.drop()

    request.addfinalizer(drop)

    return coll


@pytest.fixture()
def coll_write_ini_section(coll, ini_test):
    write_ini_section(ini_test, "general")
    return coll


def test_db_empty(db):
    assert len(db.list_collection_names()) == 0


def test_insert(coll):
    coll.insert_one({"test": "test"})
    item = coll.find_one({"test": "test"})
    assert item != None


def test_drop(coll):
    coll.insert_one({"test": "test"})
    coll.drop()
    item = coll.find_one({"test": "test"})
    assert item == None


def test_upsert(coll):
    coll.update_one({"test": "test"}, {"$set": {"other": "other"}}, upsert=True)
    item = coll.find_one({"test": "test"})
    assert item != None


def test_write_ini_section(coll_write_ini_section):
    item = coll_write_ini_section.find_one({"section": "general"})
    assert item != None


def test_write_ini_section_feed_same_ini_no_change(coll_write_ini_section, ini_test):
    item = coll_write_ini_section.find_one({"section": "general"})
    write_ini_section(ini_test, "general")
    item2 = coll_write_ini_section.find_one({"section": "general"})
    assert item == item2


def test_write_ini_section_override_answer_no(
    monkeypatch, coll_write_ini_section, ini_test_change
):
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    with pytest.raises(OpensoundscapeAttemptOverrideINISection):
        write_ini_section(ini_test_change, "general")


def test_write_ini_section_override_answer_default(
    monkeypatch, coll_write_ini_section, ini_test_change
):
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    with pytest.raises(OpensoundscapeAttemptOverrideINISection):
        write_ini_section(ini_test_change, "general")


def test_write_ini_section_override_answer_yes(
    monkeypatch, coll_write_ini_section, ini_test_change
):
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    write_ini_section(ini_test_change, "general")
    item = coll_write_ini_section.find_one({"section": "general"})
    assert item["db_sparse"] == "False"
