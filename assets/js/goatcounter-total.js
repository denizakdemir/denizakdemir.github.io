document.addEventListener('DOMContentLoaded', () => {
  const visitorElement = document.querySelector('.gc-visitors-count');
  
  if (visitorElement) {
    // Use GoatCounter's public API to get site statistics
    const siteId = 'denizakdemir';
    const url = `https://${siteId}.goatcounter.com/counter/TOTAL.json`;
    
    fetch(url)
      .then(response => response.json())
      .then(data => {
        if (data && data.count) {
          const count = data.count.replace(/\s/g, '');
          visitorElement.innerText = `${new Intl.NumberFormat().format(count)} visitors`;
        }
      })
      .catch(error => {
        console.error('Error fetching visitor count:', error);
        // Try fallback to show at least something
        visitorElement.innerText = 'Tracking visitors';
      });
  }
});