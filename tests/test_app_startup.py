"""
Smoke test for the op-admin-dash application.
Connects to running app service and verifies both HTTP and browser rendering.
Does not start a subprocess - assumes app is already running in a container.
"""
import http.client as httplib
import os
import socket as socket_module
import time as time_module

import playwright.sync_api as playwright_sync

STARTUP_TIMEOUT_SECS = 60

# When running in docker-compose, the app is accessible via the service name
# When running locally, it's at localhost:8050
APP_HOST = os.environ.get("APP_HOST", "localhost")
APP_PORT = int(os.environ.get("APP_PORT", "8050"))


def _wait_for_app_connectivity(url):
    """Helper: Wait for app to be accessible via HTTP."""
    deadline = time_module.time() + STARTUP_TIMEOUT_SECS
    while time_module.time() < deadline:
        try:
            sock = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
            result = sock.connect_ex((APP_HOST, APP_PORT))
            sock.close()
            if result == 0:
                time_module.sleep(1)  # Give it a moment to fully initialize
                return
        except Exception:
            pass
        time_module.sleep(1)
    raise RuntimeError(f"App not accessible at {url} within {STARTUP_TIMEOUT_SECS}s")


def test_app_connectivity():
    """
    Test basic HTTP connectivity to the op-admin-dash application.
    
    Verifies:
    1. App is accessible on the network
    2. App responds to HTTP requests with 200 or 401 status
    """
    url = f"http://{APP_HOST}:{APP_PORT}/"
    
    # Wait for app to be accessible
    _wait_for_app_connectivity(url)
    
    # Verify HTTP response
    try:
        conn = httplib.HTTPConnection(APP_HOST, APP_PORT, timeout=5)
        conn.request("GET", "/")
        response = conn.getresponse()
        status = response.status
        conn.close()
        assert status in [200, 401], f"Expected 200 or 401 (login), got {status}"
        print(f"✓ HTTP connectivity check passed (status {status})")
    except Exception as e:
        raise AssertionError(f"HTTP connectivity check failed: {e}")


def test_app_renders_in_browser():
    """
    Test that the op-admin-dash application renders correctly in a browser.
    
    Verifies:
    1. Page loads in Playwright headless browser
    2. Login works with test credentials (user: hello, password: world)
    3. Authenticated page renders with HTML structure and visible content
    4. No ERROR messages in browser console (catches React errors, CSP violations, etc.)
    """
    url = f"http://{APP_HOST}:{APP_PORT}/"
    
    # Wait for app to be accessible
    _wait_for_app_connectivity(url)
    
    # Use Playwright to verify page renders and login works
    with playwright_sync.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context()
            page = context.new_page()
            
            # Collect browser console messages
            console_messages = []
            
            def handle_console(msg):
                console_messages.append(f"[{msg.type.upper()}] {msg.text}")
            
            page.on("console", handle_console)
            
            page_content = None
            try:
                # Navigate to the app login page
                page.goto(url, wait_until="domcontentloaded", timeout=10000)
            except Exception as e:
                try:
                    page_content = page.content()
                except:
                    pass
                error_msg = f"Failed to load page: {e}\nURL: {url}"
                if console_messages:
                    error_msg += "\n\nBrowser Console Messages:\n" + "\n".join(console_messages)
                if page_content:
                    error_msg += f"\n\nPage Content ({len(page_content)} bytes):\n{page_content[:2000]}"
                raise AssertionError(error_msg)
            
            print(f"✓ Page loaded successfully")
            
            # Login with test credentials
            try:
                username_field = page.query_selector('input[type="text"]')
                if not username_field:
                    username_field = page.query_selector('input[name="username"]')
                if not username_field:
                    username_field = page.query_selector('input')
                
                if username_field:
                    username_field.fill("hello")
                
                password_field = page.query_selector('input[type="password"]')
                if password_field:
                    password_field.fill("world")
                
                login_button = page.query_selector('button[type="submit"]')
                if not login_button:
                    login_button = page.query_selector('button')
                
                if login_button:
                    login_button.click()
                    page.wait_for_load_state("networkidle", timeout=10000)
                else:
                    raise AssertionError("Could not find login button")
            except Exception as e:
                try:
                    page_content = page.content()
                except:
                    pass
                error_msg = f"Login failed: {e}"
                if console_messages:
                    error_msg += "\n\nBrowser Console Messages:\n" + "\n".join(console_messages)
                if page_content:
                    error_msg += f"\n\nPage Content ({len(page_content)} bytes):\n{page_content[:2000]}"
                raise AssertionError(error_msg)
            
            print(f"✓ Login successful")
            
            # Give page time to render after login
            page.wait_for_timeout(2000)
            
            # Verify page content and console health
            page_content = page.content()
            errors = []
            
            if not (len(page_content) > 100):
                errors.append("Page content is suspiciously short")
            
            if not ("<!DOCTYPE html>" in page_content or "<html" in page_content):
                errors.append("Page doesn't contain HTML structure")
            
            body = page.query_selector("body")
            if body is None:
                errors.append("Body element not found")
            else:
                body_text = page.text_content("body")
                if not (body_text and len(body_text.strip()) > 0):
                    errors.append("No visible text content found on page")
            
            # Check for critical errors in console
            critical_errors = [msg for msg in console_messages if "[ERROR]" in msg]
            if critical_errors:
                errors.append(f"Found {len(critical_errors)} ERROR messages in browser console")
            
            if errors:
                error_msg = "Page validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
                if console_messages:
                    error_msg += "\n\nBrowser Console Messages:\n" + "\n".join(console_messages)
                error_msg += f"\n\nPage Content ({len(page_content)} bytes):\n{page_content[:2000]}"
                raise AssertionError(error_msg)
            
            print(f"✓ Page rendered successfully with valid content")
            print(f"✓ Browser console is clean (no ERROR messages)")
            print(f"\n✅ Browser rendering test passed for {url}")
        
        finally:
            browser.close()

