"""
Patch for ADK CLI to fix MCP session cleanup issue.

The bug: runner.close() is called inside run_interactively(), but then
run_cli() tries to access session_service.get_session() after that,
causing a cancel scope violation during cleanup.

The fix: Return the runner from run_interactively() and close it AFTER
session saving is complete.
"""

import sys
from pathlib import Path


def apply_patch():
  """Apply the patch to the ADK cli.py file."""

  # Find the ADK installation
  import google.adk

  adk_path = Path(google.adk.__file__).parent
  cli_file = adk_path / "cli" / "cli.py"

  if not cli_file.exists():
    print(f"Error: Could not find {cli_file}")
    return False

  print(f"Patching {cli_file}")

  # Read the file
  content = cli_file.read_text()

  # Check if already patched
  if "# PATCHED: MCP cleanup fix" in content:
    print("Already patched!")
    return True

  # Patch 1: Modify run_interactively to return runner instead of closing it
  old_code_1 = """  await runner.close()


async def run_cli("""

  new_code_1 = """  # PATCHED: MCP cleanup fix - return runner instead of closing
  return runner


async def run_cli("""

  if old_code_1 not in content:
    print("Error: Could not find code to patch (section 1)")
    return False

  content = content.replace(old_code_1, new_code_1)

  # Patch 2: Update run_cli to close runner after session saving
  old_code_2 = """    await run_interactively(
        agent_or_app,
        artifact_service,
        session,
        session_service,
        credential_service,
    )

  if save_session:"""

  new_code_2 = """    # PATCHED: MCP cleanup fix - get runner to close it later
    runner = await run_interactively(
        agent_or_app,
        artifact_service,
        session,
        session_service,
        credential_service,
    )

  if save_session:"""

  if old_code_2 not in content:
    print("Error: Could not find code to patch (section 2)")
    return False

  content = content.replace(old_code_2, new_code_2)

  # Patch 3: Close runner after session saving
  # Need to add runner close at the end of run_cli
  # Find the end of the save_session block and add runner close
  if "print('Session saved to', session_path)" in content:
    # Add runner close after the entire run_cli function logic
    old_end = "    print('Session saved to', session_path)\n"
    new_end = """    print('Session saved to', session_path)
  
  # PATCHED: MCP cleanup fix - close runner after session is saved
  if 'runner' in locals():
    await runner.close()
"""
    content = content.replace(old_end, new_end)

  # Also handle the input_file case
  old_code_4 = """      async for event in agen:
        if event.content and event.content.parts:
          if text := ''.join(part.text or '' for part in event.content.parts):
            click.echo(f'[{event.author}]: {text}')
  return session"""

  new_code_4 = """      async for event in agen:
        if event.content and event.content.parts:
          if text := ''.join(part.text or '' for part in event.content.parts):
            click.echo(f'[{event.author}]: {text}')
  # PATCHED: MCP cleanup fix - close runner before returning
  await runner.close()
  return session"""

  if old_code_4 in content:
    content = content.replace(old_code_4, new_code_4)

  # Backup the original
  backup_file = cli_file.with_suffix(".py.backup")
  if not backup_file.exists():
    backup_file.write_text(cli_file.read_text())
    print(f"Backup created: {backup_file}")

  # Write the patched version
  cli_file.write_text(content)
  print("Patch applied successfully!")
  return True


def revert_patch():
  """Revert the patch."""
  import google.adk

  adk_path = Path(google.adk.__file__).parent
  cli_file = adk_path / "cli" / "cli.py"
  backup_file = cli_file.with_suffix(".py.backup")

  if not backup_file.exists():
    print("No backup found!")
    return False

  cli_file.write_text(backup_file.read_text())
  print("Patch reverted!")
  return True


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "revert":
    revert_patch()
  else:
    apply_patch()
