"""Use the installed Colab SDK, retaining the kernel and collecting errors in-cell."""
from pathlib import Path


def mount_code():
    diagnostics = Path(__file__).with_name('drive_diagnostics.py').read_text()
    return (
        'import json, os\nfrom google.colab import drive\n'
        'glm_mount_error = None\n'
        'try:\n    drive.mount("/content/drive")\n'
        'except Exception as exc:\n'
        '    glm_mount_error = type(exc).__name__\n'
        '    print("[glm53-a100] Driveマウント失敗。診断を取得します。", flush=True)\n'
        'glm_diagnostics = {"__name__": "glm_drive_diagnostics"}\n'
        f'exec(compile({diagnostics!r}, "drive-diagnostics", "exec"), glm_diagnostics)\n'
        'glm_info = glm_diagnostics["collect"]()\n'
        'glm_info["mount_error_type"] = glm_mount_error\n'
        'print("GLM_DRIVE_RESULT=" + json.dumps(glm_info), flush=True)\n'
    )


def attached_runtime(factory, state, *args, **kwargs):
    def remember_kernel(kernel_id):
        current.kernel_id = kernel_id
        state.store.add(current)

    def remember_session(session_id):
        current.session_id = session_id
        state.store.add(current)

    current = state.get_session(kwargs['session_name'])
    kwargs.update(kernel_id=current.kernel_id, session_id=current.session_id,
                  on_kernel_started=remember_kernel, on_session_started=remember_session)
    return factory(*args, **kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--session', required=True)
    args = parser.parse_args()
    from colab_cli.commands import automation
    from colab_cli.common import state
    from colab_cli.cli import app
    # The installed automation command starts another kernel instead of reusing
    # the IDs cached by exec. Keep its auth handler, but attach to our kernel.
    original = automation.ColabRuntime
    original_automation = automation.run_automation
    automation.ColabRuntime = lambda *a, **kw: attached_runtime(original, state, *a, **kw)

    def with_diagnostics(name, op, code, **kwargs):
        kwargs['timeout'] = 300
        return original_automation(name, op, mount_code(), **kwargs)

    automation.run_automation = with_diagnostics
    try:
        # Run the CLI callback as well: this retains its authentication defaults,
        # logging and configuration initialization.
        app(args=['drivemount', '-s', args.session])
    finally:
        automation.ColabRuntime = original
        automation.run_automation = original_automation


if __name__ == '__main__':
    main()
