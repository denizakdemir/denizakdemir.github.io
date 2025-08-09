// ===========================
// SAS Programming Tutorial - Main JavaScript
// ===========================

// --- DOM Content Loaded ---
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeProgressIndicator();
    initializeBackToTop();
    initializeCodeCopy();
    initializeNewsletter();
    initializeSmoothScroll();
    initializeProgressTracking();
});

// --- Navigation Menu ---
function initializeNavigation() {
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            
            // Animate hamburger menu
            const spans = navToggle.querySelectorAll('span');
            spans[0].style.transform = navMenu.classList.contains('active') 
                ? 'rotate(-45deg) translate(-5px, 6px)' : '';
            spans[1].style.opacity = navMenu.classList.contains('active') ? '0' : '1';
            spans[2].style.transform = navMenu.classList.contains('active') 
                ? 'rotate(45deg) translate(-5px, -6px)' : '';
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', function(e) {
            if (!navToggle.contains(e.target) && !navMenu.contains(e.target)) {
                navMenu.classList.remove('active');
                resetHamburgerMenu();
            }
        });
        
        // Close menu when clicking on a link
        navMenu.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navMenu.classList.remove('active');
                resetHamburgerMenu();
            });
        });
    }
}

function resetHamburgerMenu() {
    const navToggle = document.querySelector('.nav-toggle');
    const spans = navToggle.querySelectorAll('span');
    spans[0].style.transform = '';
    spans[1].style.opacity = '1';
    spans[2].style.transform = '';
}

// --- Progress Indicator ---
function initializeProgressIndicator() {
    const progressIndicator = document.getElementById('progress-indicator');
    
    if (progressIndicator) {
        window.addEventListener('scroll', function() {
            const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
            const scrolled = (window.scrollY / windowHeight) * 100;
            progressIndicator.style.width = scrolled + '%';
        });
    }
}

// --- Back to Top Button ---
function initializeBackToTop() {
    const backToTopButton = document.getElementById('back-to-top');
    
    if (backToTopButton) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 300) {
                backToTopButton.classList.add('show');
            } else {
                backToTopButton.classList.remove('show');
            }
        });
        
        backToTopButton.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }
}

// --- Code Copy Functionality ---
function initializeCodeCopy() {
    const copyButtons = document.querySelectorAll('.copy-btn');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const codeBlock = this.parentElement.nextElementSibling.querySelector('code');
            const textToCopy = codeBlock.textContent;
            
            navigator.clipboard.writeText(textToCopy).then(() => {
                // Change button text
                const originalText = this.textContent;
                this.textContent = 'Copied!';
                this.classList.add('copied');
                
                // Reset after 2 seconds
                setTimeout(() => {
                    this.textContent = originalText;
                    this.classList.remove('copied');
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy:', err);
            });
        });
    });
}

// --- Newsletter Form ---
function initializeNewsletter() {
    const newsletterForm = document.getElementById('newsletter-form');
    
    if (newsletterForm) {
        newsletterForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const emailInput = this.querySelector('input[type="email"]');
            const email = emailInput.value;
            
            // Here you would normally send the email to your server
            // For now, just show a success message
            showNotification('Thank you for subscribing! You\'ll receive updates about new tutorials.');
            
            // Clear the form
            emailInput.value = '';
        });
    }
}

// --- Smooth Scrolling ---
function initializeSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                const offset = 80; // Account for fixed navbar
                const targetPosition = targetElement.offsetTop - offset;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// --- Progress Tracking ---
function initializeProgressTracking() {
    // Check if we're on a tutorial page
    if (document.body.classList.contains('tutorial-page')) {
        const tutorialId = document.body.dataset.tutorialId;
        
        if (tutorialId) {
            // Check if tutorial is already completed
            const isCompleted = localStorage.getItem(`sas-tutorial-${tutorialId}`) === 'completed';
            
            if (isCompleted) {
                markTutorialComplete(tutorialId);
            }
            
            // Add complete button functionality
            const completeBtn = document.getElementById('complete-tutorial');
            if (completeBtn) {
                completeBtn.addEventListener('click', function() {
                    localStorage.setItem(`sas-tutorial-${tutorialId}`, 'completed');
                    markTutorialComplete(tutorialId);
                    updateProgressDisplay();
                    showNotification('Tutorial marked as complete! Great job!');
                });
            }
        }
    }
    
    // Update progress display on main page
    updateProgressDisplay();
}

function markTutorialComplete(tutorialId) {
    // Add visual indicator on tutorial page
    const progressBadge = document.querySelector('.progress-badge');
    if (progressBadge) {
        progressBadge.classList.add('completed');
        progressBadge.textContent = 'Completed';
    }
}

function updateProgressDisplay() {
    // Update progress stats on main page
    const totalTutorials = 13;
    let completedCount = 0;
    
    for (let i = 1; i <= totalTutorials; i++) {
        if (localStorage.getItem(`sas-tutorial-${i}`) === 'completed') {
            completedCount++;
            
            // Update tutorial card on main page
            const card = document.querySelector(`[data-tutorial="${i}"]`);
            if (card) {
                card.classList.add('completed');
            }
        }
    }
    
    // Update progress percentage
    const progressPercent = Math.round((completedCount / totalTutorials) * 100);
    const progressDisplay = document.getElementById('progress-display');
    if (progressDisplay) {
        progressDisplay.textContent = `${progressPercent}% Complete (${completedCount}/${totalTutorials} tutorials)`;
    }
}

// --- Notification System ---
function showNotification(message, type = 'success') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Trigger animation
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

// --- Search Functionality ---
function initializeSearch() {
    const searchInput = document.getElementById('tutorial-search');
    const tutorialCards = document.querySelectorAll('.tutorial-card');
    
    if (searchInput && tutorialCards.length > 0) {
        searchInput.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            
            tutorialCards.forEach(card => {
                const title = card.querySelector('.card-title').textContent.toLowerCase();
                const description = card.querySelector('.card-description').textContent.toLowerCase();
                const topics = Array.from(card.querySelectorAll('.card-topics li'))
                    .map(li => li.textContent.toLowerCase())
                    .join(' ');
                
                const searchContent = `${title} ${description} ${topics}`;
                
                if (searchContent.includes(searchTerm)) {
                    card.style.display = '';
                } else {
                    card.style.display = 'none';
                }
            });
            
            // Show/hide no results message
            const visibleCards = document.querySelectorAll('.tutorial-card:not([style*="display: none"])');
            const noResultsMsg = document.getElementById('no-results');
            
            if (visibleCards.length === 0 && noResultsMsg) {
                noResultsMsg.style.display = 'block';
            } else if (noResultsMsg) {
                noResultsMsg.style.display = 'none';
            }
        });
    }
}

// --- Active Section Highlighting ---
function initializeActiveSectionHighlighting() {
    if (document.body.classList.contains('tutorial-page')) {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.tutorial-nav a');
        
        window.addEventListener('scroll', () => {
            let current = '';
            
            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                const sectionHeight = section.clientHeight;
                
                if (scrollY >= sectionTop - 100) {
                    current = section.getAttribute('id');
                }
            });
            
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href').slice(1) === current) {
                    link.classList.add('active');
                }
            });
        });
    }
}

// --- Keyboard Shortcuts ---
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('tutorial-search');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // ESC to close mobile menu
        if (e.key === 'Escape') {
            const navMenu = document.querySelector('.nav-menu');
            if (navMenu && navMenu.classList.contains('active')) {
                navMenu.classList.remove('active');
                resetHamburgerMenu();
            }
        }
    });
}

// --- Lazy Loading Images ---
function initializeLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
}

// --- Initialize Additional Features ---
window.addEventListener('load', function() {
    initializeSearch();
    initializeActiveSectionHighlighting();
    initializeKeyboardShortcuts();
    initializeLazyLoading();
});

// --- Utility Functions ---
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// --- CSS Notification Styles (to be added dynamically) ---
const notificationStyles = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: #2ca02c;
        color: white;
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transform: translateX(400px);
        transition: transform 0.3s ease;
        z-index: 10000;
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification-error {
        background: #d62728;
    }
    
    .notification-warning {
        background: #ff9800;
    }
`;

// Add notification styles to page
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);