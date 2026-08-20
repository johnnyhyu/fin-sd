import pytest

from big_finance_harness.tools.base import ToolError
from big_finance_harness.tools.python_exec import PythonExecTool


@pytest.mark.asyncio
async def test_python_exec_basic_arithmetic():
    tool = PythonExecTool()
    result = await tool.run({"code": "print(2 + 2)"})
    assert "exit_code: 0" in result
    assert "4" in result


@pytest.mark.asyncio
async def test_python_exec_captures_stderr():
    tool = PythonExecTool()
    result = await tool.run({"code": "import sys; sys.stderr.write('boom\\n'); sys.exit(2)"})
    assert "exit_code: 2" in result
    assert "boom" in result


@pytest.mark.asyncio
async def test_python_exec_timeout():
    tool = PythonExecTool(timeout_s=0.5)
    with pytest.raises(ToolError, match="timed out"):
        await tool.run({"code": "import time; time.sleep(5)"})


@pytest.mark.asyncio
async def test_python_exec_rejects_empty():
    tool = PythonExecTool()
    with pytest.raises(ToolError):
        await tool.run({"code": "   "})


@pytest.mark.asyncio
async def test_python_exec_reports_unparseable_tool_arguments():
    """When a provider hands back tool-call arguments that aren't valid JSON, the client
    stores the raw string under `_unparsed_arguments`. Reporting "code is required"
    tells the model it forgot an argument it actually sent."""
    tool = PythonExecTool()
    with pytest.raises(ToolError, match="not valid JSON"):
        await tool.run({"_unparsed_arguments": '{"code": "print(1)'})


@pytest.mark.asyncio
async def test_python_exec_missing_code_lists_the_keys_it_got():
    tool = PythonExecTool()
    with pytest.raises(ToolError, match=r"Received argument keys: \['source'\]"):
        await tool.run({"source": "print(1)"})


@pytest.mark.asyncio
async def test_python_exec_does_not_strip_code():
    """`strip=True` would rewrite a snippet whose leading indentation is significant."""
    tool = PythonExecTool()
    result = await tool.run({"code": "\nprint(2 + 2)\n"})
    assert "exit_code: 0" in result
    assert "4" in result
