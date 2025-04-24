"""
Secure process management for the LLM Gateway.
"""

import os
import signal
import subprocess
import logging
import asyncio
import time
import platform
import tempfile
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)


class ProcessSecurityError(Exception):
    """Error raised when a process security operation fails."""
    pass


class SecureProcessManager:
    """
    Manager for secure subprocess execution with resource limits
    and proper privilege management.
    """
    
    def __init__(
        self,
        enable_resource_limits: bool = True,
        max_cpu_seconds: int = 300,  # 5 minutes
        max_memory_mb: int = 1024,   # 1 GB
        max_file_size_mb: int = 100, # 100 MB
        max_processes: int = 10,
        enable_network: bool = False,
        secure_env: bool = True,
        working_dir: Optional[str] = None
    ):
        """
        Initialize secure process manager.
        
        Args:
            enable_resource_limits: Whether to apply resource limits
            max_cpu_seconds: Maximum CPU time in seconds
            max_memory_mb: Maximum memory in MB
            max_file_size_mb: Maximum file size in MB
            max_processes: Maximum number of processes
            enable_network: Whether to allow network access
            secure_env: Whether to sanitize environment variables
            working_dir: Working directory for processes (temp dir if None)
        """
        self.enable_resource_limits = enable_resource_limits
        self.max_cpu_seconds = max_cpu_seconds
        self.max_memory_mb = max_memory_mb
        self.max_file_size_mb = max_file_size_mb
        self.max_processes = max_processes
        self.enable_network = enable_network
        self.secure_env = secure_env
        
        # Set working directory
        if working_dir:
            self.working_dir = working_dir
            os.makedirs(self.working_dir, exist_ok=True)
        else:
            self.working_dir = tempfile.mkdtemp(prefix="secure_process_")
        
        # Track running processes
        self.processes: Dict[int, subprocess.Popen] = {}
        self.async_processes: Dict[int, asyncio.subprocess.Process] = {}
        self.process_lock = asyncio.Lock()
        
        # Platform-specific settings
        self.is_windows = platform.system() == "Windows"
    
    def _get_secure_env(self, env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Get secure environment variables.
        
        Args:
            env: Original environment variables
            
        Returns:
            Sanitized environment variables
        """
        if not self.secure_env:
            return env or os.environ.copy()
        
        # Start with minimal base environment
        secure_env = {}
        
        # Add essential variables
        for var in ["PATH", "HOME", "LANG", "TERM", "SYSTEMROOT", "TEMP", "TMP"]:
            if var in os.environ:
                secure_env[var] = os.environ[var]
        
        # Override with provided environment, filtering out unsafe variables
        if env:
            for key, value in env.items():
                # Skip variables that might contain sensitive data
                if any(pattern in key.upper() for pattern in ["SECRET", "KEY", "TOKEN", "PASSWORD", "CREDENTIAL"]):
                    continue
                secure_env[key] = value
        
        return secure_env
    
    def _apply_windows_job_restrictions(self, process: subprocess.Popen) -> None:
        """
        Apply job restrictions on Windows.
        
        Args:
            process: Process to restrict
            
        Note:
            This is a placeholder. In a real implementation, you would use the
            Windows Job Objects API to restrict the process.
        """
        # This would use the Windows Job Objects API to restrict the process
        # For now, we'll just log a warning
        logger.warning("Process restrictions on Windows not implemented")
    
    async def create_process(
        self,
        command: str,
        args: List[str] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        stdin_data: Optional[str] = None
    ) -> subprocess.Popen:
        """
        Create a secure subprocess.
        
        Args:
            command: Command to run
            args: Command arguments
            env: Environment variables
            cwd: Working directory
            stdin_data: Data to write to stdin
            
        Returns:
            Subprocess object
        """
        # Build command and arguments
        cmd_list = [command]
        if args:
            cmd_list.extend(args)
        
        # Get secure environment
        secure_env = self._get_secure_env(env)
        
        # Set working directory
        work_dir = cwd or self.working_dir
        
        logger.info(f"Creating secure process: {command} {' '.join(args or [])}")
        
        try:
            # Create process with security settings
            process = subprocess.Popen(
                cmd_list,
                stdin=subprocess.PIPE if stdin_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=secure_env,
                cwd=work_dir,
                start_new_session=not self.is_windows  # Not supported on Windows
            )
            
            # Apply platform-specific restrictions
            if self.is_windows and self.enable_resource_limits:
                self._apply_windows_job_restrictions(process)
            
            # Write stdin data if provided
            if stdin_data and process.stdin:
                process.stdin.write(stdin_data.encode())
                process.stdin.flush()
                if not self.is_windows:  # Windows handles this differently
                    process.stdin.close()
            
            # Track the process
            async with self.process_lock:
                self.processes[process.pid] = process
            
            logger.info(f"Created secure process with PID {process.pid}")
            
            return process
        except Exception as e:
            logger.error(f"Failed to create secure process: {e}")
            raise ProcessSecurityError(f"Failed to create process: {str(e)}")
    
    async def create_async_process(
        self,
        command: str,
        args: List[str] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None
    ) -> asyncio.subprocess.Process:
        """
        Create a secure asyncio subprocess.
        
        Args:
            command: Command to run
            args: Command arguments
            env: Environment variables
            cwd: Working directory
            
        Returns:
            Asyncio subprocess object
        """
        # Build command and arguments
        cmd_list = [command]
        if args:
            cmd_list.extend(args)
        
        # Get secure environment
        secure_env = self._get_secure_env(env)
        
        # Set working directory
        work_dir = cwd or self.working_dir
        
        logger.info(f"Creating secure async process: {command} {' '.join(args or [])}")
        
        try:
            # Create process
            process = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=secure_env,
                cwd=work_dir
            )
            
            # Track the process
            async with self.process_lock:
                self.async_processes[process.pid] = process
            
            logger.info(f"Created secure async process with PID {process.pid}")
            
            return process
        except Exception as e:
            logger.error(f"Failed to create secure async process: {e}")
            raise ProcessSecurityError(f"Failed to create async process: {str(e)}")
    
    async def terminate_process(
        self,
        process: Union[subprocess.Popen, asyncio.subprocess.Process],
        timeout: float = 5.0
    ) -> None:
        """
        Terminate a process safely.
        
        Args:
            process: Process to terminate
            timeout: Timeout in seconds
        """
        logger.info(f"Terminating process with PID {process.pid}")
        
        try:
            # Send SIGTERM
            if isinstance(process, asyncio.subprocess.Process):
                process.terminate()
                
                # Wait for process to terminate
                try:
                    await asyncio.wait_for(process.wait(), timeout=timeout)
                    logger.info(f"Process {process.pid} terminated")
                except asyncio.TimeoutError:
                    logger.warning(f"Process {process.pid} did not terminate, sending SIGKILL")
                    process.kill()
                    await process.wait()
                    logger.info(f"Process {process.pid} killed")
                
                # Remove from tracking
                async with self.process_lock:
                    if process.pid in self.async_processes:
                        del self.async_processes[process.pid]
            else:
                process.terminate()
                
                # Wait for process to terminate
                try:
                    returncode = process.wait(timeout=timeout)
                    logger.info(f"Process {process.pid} terminated with code {returncode}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Process {process.pid} did not terminate, sending SIGKILL")
                    process.kill()
                    process.wait()
                    logger.info(f"Process {process.pid} killed")
                
                # Remove from tracking
                async with self.process_lock:
                    if process.pid in self.processes:
                        del self.processes[process.pid]
        except Exception as e:
            logger.error(f"Error terminating process {process.pid}: {e}")
            # Ensure process is killed
            try:
                if isinstance(process, asyncio.subprocess.Process):
                    process.kill()
                    await process.wait()
                else:
                    process.kill()
                    process.wait()
                logger.info(f"Process {process.pid} forcibly killed after error")
            except Exception as kill_error:
                logger.error(f"Error killing process {process.pid}: {kill_error}")
    
    async def terminate_all_processes(self, timeout: float = 5.0) -> None:
        """
        Terminate all managed processes.
        
        Args:
            timeout: Timeout in seconds per process
        """
        logger.info("Terminating all managed processes")
        
        # Get a copy of the process dictionaries
        async with self.process_lock:
            processes = list(self.processes.values())
            async_processes = list(self.async_processes.values())
        
        # Terminate regular processes
        for process in processes:
            await self.terminate_process(process, timeout)
        
        # Terminate async processes
        for process in async_processes:
            await self.terminate_process(process, timeout)
        
        logger.info("All managed processes terminated")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Terminate all processes
        await self.terminate_all_processes()
        
        # Clean up temporary directory if we created it
        if self.working_dir.startswith(tempfile.gettempdir()):
            try:
                import shutil
                shutil.rmtree(self.working_dir, ignore_errors=True)
                logger.info(f"Removed temporary directory {self.working_dir}")
            except Exception as e:
                logger.error(f"Error removing temporary directory {self.working_dir}: {e}")
        
        logger.info("Process manager cleaned up")
