import requests
import logging
import random
import asyncio
import aiohttp
import subprocess
import time
import sys
import ctypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('proxy_manager')

def is_admin():
    """
    Check if the script is running with administrative privileges
    
    :return: Boolean indicating admin status
    """
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False

class VPNManager:
    def __init__(self, vpn_service='nordvpn'):
        """
        Initialize VPN manager to rotate IP addresses
        
        :param vpn_service: VPN service to use (default: NordVPN)
        """
        self.vpn_service = vpn_service
        self.current_connection = None
    
    def connect(self):
        """
        Connect to a new VPN server
        """
        try:
            # Disconnect existing connection if any
            self.disconnect()
            
            # Connect to a random server
            connect_cmd = self._get_connect_command()
            success, stdout, stderr = self.run_command(connect_cmd)
            
            if success:
                logger.info(f"Connected to new VPN server")
                self.current_connection = time.time()
                return True
            else:
                logger.error(f"VPN connection failed: {stderr}")
                return False
        
        except Exception as e:
            logger.error(f"Error connecting to VPN: {e}")
            return False
    
    def disconnect(self):
        """
        Disconnect from current VPN server
        """
        try:
            disconnect_cmd = self._get_disconnect_command()
            success, stdout, stderr = self.run_command(disconnect_cmd)
            
            if success:
                logger.info("Disconnected from VPN")
                self.current_connection = None
                return True
            else:
                logger.warning(f"VPN disconnection failed: {stderr}")
                return False
        
        except Exception as e:
            logger.error(f"Error disconnecting from VPN: {e}")
            return False
    
    def _get_connect_command(self):
        """
        Get VPN connection command based on service
        """
        if self.vpn_service == 'nordvpn':
            return 'nordvpn connect'
        elif self.vpn_service == 'expressvpn':
            return 'expressvpn connect'
        else:
            raise ValueError(f"Unsupported VPN service: {self.vpn_service}")
    
    def _get_disconnect_command(self):
        """
        Get VPN disconnection command based on service
        """
        if self.vpn_service == 'nordvpn':
            return 'nordvpn disconnect'
        elif self.vpn_service == 'expressvpn':
            return 'expressvpn disconnect'
        else:
            raise ValueError(f"Unsupported VPN service: {self.vpn_service}")
    
    def get_current_ip(self):
        """
        Get current public IP address
        """
        try:
            response = requests.get('https://api.ipify.org')
            return response.text
        except Exception as e:
            logger.error(f"Error getting IP address: {e}")
            return None
    
    @staticmethod
    def run_command(cmd):
        """
        Run a command with robust error handling and encoding
        
        :param cmd: Command to run
        :return: Tuple of (success, output, error)
        """
        try:
            # Use universal_newlines and text to handle encoding
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True,  # Use text mode
                encoding='utf-8',  # Specify UTF-8 encoding
                errors='replace'  # Replace undecodable bytes
            )
            
            # Check if command was successful
            success = result.returncode == 0
            
            return success, result.stdout.strip(), result.stderr.strip()
        
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return False, "", str(e)

class DNSManager:
    # Comprehensive global DNS servers from various providers
    DNS_SERVERS = [
        # Google Public DNS
        '8.8.8.8', '8.8.4.4',
        
        # Cloudflare DNS
        '1.1.1.1', '1.0.0.1',
        
        # Quad9 DNS (Privacy & Security Focused)
        '9.9.9.9', '149.112.112.112',
        
        # OpenDNS
        '208.67.222.222', '208.67.220.220',
        
        # Comodo Secure DNS
        '8.26.56.26', '8.20.247.20',
        
        # Norton ConnectSafe
        '199.85.126.10', '199.85.127.10',
        
        # Level 3 DNS
        '4.2.2.1', '4.2.2.2',
        
        # Verisign Public DNS
        '64.6.64.6', '64.6.65.6',
        
        # Alternate DNS
        '76.76.19.19', '76.223.122.150',
        
        # AdGuard DNS
        '94.140.14.14', '94.140.15.15',
        
        # CleanBrowsing DNS
        '185.228.168.9', '185.228.169.9',
        
        # Yandex DNS
        '77.88.8.8', '77.88.8.1',
        
        # Control D DNS
        '76.76.2.0', '76.76.10.0',
        
        # Neustar DNS
        '156.154.70.1', '156.154.71.1',
        
        # International DNS Servers
        '185.121.177.177',  # Iranian DNS
        '178.22.122.100',   # Iranian DNS
        '91.229.210.190',   # Romanian DNS
        '89.107.194.2',     # Bulgarian DNS
        '195.5.195.195',    # Swiss DNS
        '194.145.240.2',    # Turkish DNS
    ]

    @staticmethod
    def run_command(cmd, require_admin=False):
        """
        Run a command with robust error handling and encoding
        
        :param cmd: Command to run
        :param require_admin: If True, will log warning if not running as admin
        :return: Tuple of (success, output, error)
        """
        try:
            # Check for admin privileges if required
            if require_admin and not is_admin():
                logger.warning("🚨 Administrative privileges required for this operation! 🔒")
                return False, "", "Requires admin privileges"
            
            # Use universal_newlines and text to handle encoding
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True,  # Use text mode
                encoding='utf-8',  # Specify UTF-8 encoding
                errors='replace'  # Replace undecodable bytes
            )
            
            # Check if command was successful
            success = result.returncode == 0
            
            return success, result.stdout.strip(), result.stderr.strip()
        
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return False, "", str(e)

    @staticmethod
    def change_dns(interface='Ethernet'):
        """
        Change DNS servers using Windows command-line tools
        
        :param interface: Network interface name
        """
        try:
            # Select two random DNS servers
            primary_dns = random.choice(DNSManager.DNS_SERVERS)
            secondary_dns = random.choice([dns for dns in DNSManager.DNS_SERVERS if dns != primary_dns])
            
            # Commands to set DNS
            dns_commands = [
                f'netsh interface ip set dns "{interface}" static {primary_dns}',
                f'netsh interface ip add dns "{interface}" {secondary_dns} index=2'
            ]
            
            # Track success
            all_commands_successful = True
            
            for cmd in dns_commands:
                success, stdout, stderr = DNSManager.run_command(cmd, require_admin=True)
                if not success:
                    logger.warning(f"🚫 DNS command failed: {cmd}")
                    logger.warning(f"📝 Stdout: {stdout}")
                    logger.warning(f"❌ Stderr: {stderr}")
                    all_commands_successful = False
            
            if all_commands_successful:
                logger.info(f"🌐 DNS Changed: {primary_dns}, {secondary_dns} 🔄")
                return primary_dns, secondary_dns
            else:
                logger.warning("⚠️ Not all DNS commands were successful")
                return None, None
        
        except Exception as e:
            logger.error(f"❌ DNS Change Failed: {e}")
            return None, None

class IPManager:
    @staticmethod
    def change_ip_and_dns(interface='Ethernet'):
        """
        Change IP and DNS using Windows command-line tools
        
        :param interface: Network interface name
        """
        try:
            # Use the robust run_command method from DNSManager
            run_cmd = DNSManager.run_command
            
            # Flush DNS cache
            success, stdout, stderr = run_cmd('ipconfig /flushdns', require_admin=True)
            
            # Release current IP
            success, stdout, stderr = run_cmd(f'ipconfig /release "{interface}"', require_admin=True)
            
            # Renew IP
            success, stdout, stderr = run_cmd(f'ipconfig /renew "{interface}"', require_admin=True)
            
            logger.info(f"🌐 IP Configuration Renewed for {interface} 🔄")
            return True
        
        except Exception as e:
            logger.error(f"❌ IP Renewal Failed: {e}")
            return False

class ProxyManager:
    def __init__(self, interface='Ethernet'):
        """
        Initialize Network Configuration Manager
        
        :param interface: Network interface name
        """
        self.interface = interface
        self.dns_manager = DNSManager()
        self.ip_manager = IPManager()
        self.vpn_manager = VPNManager()
    
    def rotate_ip(self):
        """
        Rotate IP address by reconnecting VPN
        """
        try:
            old_ip = self.vpn_manager.get_current_ip()
            
            # Disconnect and reconnect
            self.vpn_manager.disconnect()
            time.sleep(2)  # Wait a moment between disconnect and connect
            self.vpn_manager.connect()
            
            # Verify IP change
            new_ip = self.vpn_manager.get_current_ip()
            
            if old_ip != new_ip:
                logger.info(f"IP Successfully Rotated: {old_ip} -> {new_ip}")
                return True
            else:
                logger.warning("IP rotation failed - same IP detected")
                return False
        
        except Exception as e:
            logger.error(f"Error rotating IP: {e}")
            return False
    
    def rotate_network(self):
        """
        Rotate network configuration
        """
        try:
            # Check if running with admin privileges
            if not is_admin():
                logger.warning("🚨 Network rotation requires administrative privileges! 🔒")
                return False
            
            # Change IP configuration
            ip_result = self.ip_manager.change_ip_and_dns(self.interface)
            
            # Change DNS servers
            primary_dns, secondary_dns = self.dns_manager.change_dns(self.interface)
            
            # Small delay to allow network reconfiguration
            time.sleep(2)
            
            return ip_result and primary_dns is not None
        
        except Exception as e:
            logger.error(f"Network rotation error: {e}")
            return False

# Global proxy manager instance
proxy_manager = ProxyManager()
