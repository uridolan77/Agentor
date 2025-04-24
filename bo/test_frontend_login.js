/**
 * Test script for the frontend login functionality.
 * This script can be run in the browser console to test the login functionality.
 */

async function testFrontendLogin() {
  console.log('Testing frontend login...');
  
  // Create form data
  const formData = new URLSearchParams();
  formData.append('username', 'admin');
  formData.append('password', 'Admin123');
  
  console.log('Form data:', formData.toString());
  
  try {
    // Make the request to the backend API
    console.log('Sending request to /auth/token');
    const response = await fetch('/auth/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData.toString(),
    });
    
    console.log('Response status:', response.status);
    console.log('Response headers:', response.headers);
    
    // Parse the response
    const data = await response.json();
    console.log('Response data:', data);
    
    if (data.access_token) {
      console.log('Login successful!');
      return true;
    } else {
      console.log('Login failed: No access token in response');
      return false;
    }
  } catch (error) {
    console.error('Error:', error);
    return false;
  }
}

// Run the test
testFrontendLogin();
