from types import SimpleNamespace
from unittest.mock import patch

import pytest
from streamlit.errors import StreamlitAuthError

from pages import system_admin, system_cost


@pytest.mark.parametrize("named", [False, True])
def test_google_login_selects_configured_provider(named):
    auth = {"google": {}} if named else {}
    with patch.object(system_admin.st, "secrets", {"auth": auth}), \
            patch.object(system_admin.st, "login") as login:
        system_admin._login_with_google()
    if named:
        login.assert_called_once_with("google")
    else:
        login.assert_called_once_with()


def test_login_configuration_error_is_not_exposed():
    state = {}
    with patch.object(system_admin.st, "secrets", {}), \
            patch.object(system_admin.st, "session_state", state), \
            patch.object(system_admin.st, "login", side_effect=StreamlitAuthError("private details")):
        system_admin._login_with_google()
    assert state == {"admin_login_error": True}


@pytest.mark.parametrize("auth_available", [False, True])
def test_admin_controls_are_locked_without_login(auth_available):
    with patch.object(system_admin, "_get_auth_state", return_value=(auth_available, False, "User")), \
            patch.object(system_admin, "login_screen") as login_screen, \
            patch.object(system_admin, "load_files") as load_files, \
            patch.object(system_admin, "clear_files") as clear_files:
        system_admin.main()
    login_screen.assert_called_once()
    load_files.assert_not_called()
    clear_files.assert_not_called()


@pytest.mark.parametrize("data", [[], [{"start_time": 1, "end_time": 2, "results": []}]])
def test_empty_usage_and_cost_results_show_warnings(data):
    with patch.object(system_cost, "get_data", return_value=data), \
            patch.object(system_cost.st, "warning") as warning, \
            patch.object(system_cost.st, "write"), \
            patch.object(system_cost.st, "pyplot") as pyplot:
        system_cost.plot_cost()
    assert warning.call_count == 2
    pyplot.assert_not_called()


def test_api_failure_is_not_treated_as_empty_data():
    with patch.object(system_cost.st, "secrets", {"OPENAI_ADMIN_KEY": "test"}), \
            patch.object(system_cost.requests, "get", return_value=SimpleNamespace(status_code=401)), \
            patch.object(system_cost.st, "error") as error:
        assert system_cost.get_data("https://example.com", {}) is None
    assert "401" in error.call_args.args[0]


def test_missing_admin_key_shows_configuration_error():
    with patch.object(system_cost.st, "secrets", {}), \
            patch.object(system_cost.st, "error") as error, \
            patch.object(system_cost.requests, "get") as get:
        assert system_cost.get_data("https://example.com", {}) is None
    error.assert_called_once()
    get.assert_not_called()


def test_populated_usage_and_cost_results_render():
    usage = [{"start_time": 1700000000, "end_time": 1700086400,
              "results": [{"input_tokens": 10, "output_tokens": 5}]}]
    costs = [{"start_time": 1700000000, "end_time": 1700086400,
              "results": [{"amount": {"value": 0.25, "currency": "usd"}}]}]
    with patch.object(system_cost, "get_data", side_effect=[usage, costs]), \
            patch.object(system_cost.st, "warning") as warning, \
            patch.object(system_cost.st, "write"), \
            patch.object(system_cost.st, "pyplot") as pyplot:
        system_cost.plot_cost()
    assert pyplot.call_count == 2
    warning.assert_not_called()
    system_cost.plt.close("all")


def test_network_failure_shows_safe_error():
    with patch.object(system_cost.st, "secrets", {"OPENAI_ADMIN_KEY": "test"}), \
            patch.object(system_cost.requests, "get", side_effect=system_cost.requests.Timeout("private")), \
            patch.object(system_cost.st, "error") as error:
        assert system_cost.get_data("https://example.com", {}) is None
    assert "private" not in error.call_args.args[0]
