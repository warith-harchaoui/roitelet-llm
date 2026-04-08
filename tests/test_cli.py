import pytest
from unittest.mock import patch, MagicMock
from cli.main import main, single_prompt, print_welcome

@patch('cli.main.argparse.ArgumentParser.parse_args')
@patch('cli.main.asyncio.run')
def test_cli_ask_command(mock_run, mock_parse_args):
    mock_args = MagicMock()
    mock_args.command = "ask"
    mock_args.prompt = "Tell me a joke"
    mock_parse_args.return_value = mock_args

    main()
    mock_run.assert_called_once()
    
@patch('cli.main.argparse.ArgumentParser.parse_args')
@patch('cli.main.asyncio.run')
def test_cli_chat_command(mock_run, mock_parse_args):
    mock_args = MagicMock()
    mock_args.command = "chat"
    mock_parse_args.return_value = mock_args

    main()
    mock_run.assert_called_once()
