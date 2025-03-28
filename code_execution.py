import modal
from typing import Optional, Tuple, List

image = modal.Image.debian_slim(python_version="3.12")
app = modal.App.lookup("safe-code-execution", create_if_missing=True)
with modal.enable_output():
    sandbox = modal.Sandbox.create(app=app, image=image)

def execute_code(code_list: List[str], timeout: Optional[int] = None) -> Tuple[List[str], List[str], List[int]]:
    """
    Execute a list of Python code statements in a Modal sandbox with optional timeout.
    Each statement is executed separately.
    
    Args:
        code_list (List[str]): List of Python code statements to execute
        timeout (int, optional): Timeout in seconds. If None, no timeout is enforced.
    
    Returns:
        Tuple[List[str], List[str], List[int]]: (list of stdout outputs, list of stderr outputs, list of return codes)
    """
    with modal.enable_output():
        sandbox = modal.Sandbox.create(app=app, image=image)
    
    try:
        all_stdout = []
        all_stderr = []
        all_return_codes = []
        
        for code in code_list:
            python_ps = sandbox.exec("python", "-c", code, timeout=timeout)
            
            try:
                python_ps.wait()
                all_stdout.append(python_ps.stdout.read())
                all_stderr.append(python_ps.stderr.read())
                all_return_codes.append(python_ps.returncode)
                
                # If any statement fails, stop execution
                if python_ps.returncode != 0:
                    break
            except modal.TimeoutError:
                return all_stdout, all_stderr + ["Execution timed out"], all_return_codes + [1]
        
        return all_stdout, all_stderr, all_return_codes
        
    finally:
        sandbox.terminate()

__all__ = ["execute_code"]
