document.addEventListener("DOMContentLoaded", function () {
    console.log("Script Loaded Successfully!");

    // Highlight active navigation link
    const currentPath = window.location.pathname;
    document.querySelectorAll(".nav-links a").forEach(link => {
        if (link.getAttribute("href") === currentPath) {
            link.style.backgroundColor = "#ffffff";
            link.style.color = "#007bff";
        }
    });

    // Form Validation for Registration & Login
    const form = document.querySelector("form");
    if (form) {
        form.addEventListener("submit", function (event) {
            let valid = true;
            const username = document.querySelector("input[name='username']");
            const password = document.querySelector("input[name='password']");
            const errorMessage = document.getElementById("error-message");

            // Clear previous errors
            if (errorMessage) errorMessage.innerText = "";

            // Username validation
            if (username.value.trim() === "" || username.value.length < 3) {
                valid = false;
                username.style.border = "2px solid red";
            } else {
                username.style.border = "2px solid green";
            }

            // Password validation
            if (password.value.length < 6) {
                valid = false;
                password.style.border = "2px solid red";
            } else {
                password.style.border = "2px solid green";
            }

            if (!valid) {
                event.preventDefault();
                if (errorMessage) {
                    errorMessage.innerText = "Please fill out the form correctly.";
                }
            }
        });
    }

    // Show/hide password toggle
    const togglePassword = document.querySelector("#toggle-password");
    if (togglePassword) {
        togglePassword.addEventListener("click", function () {
            const passwordInput = document.querySelector("input[name='password']");
            if (passwordInput.type === "password") {
                passwordInput.type = "text";
                togglePassword.innerText = "Hide";
            } else {
                passwordInput.type = "password";
                togglePassword.innerText = "Show";
            }
        });
    }

    // Smooth scrolling effect for navigation
    document.querySelectorAll(".nav-links a").forEach(anchor => {
        anchor.addEventListener("click", function (event) {
            event.preventDefault();
            const targetId = this.getAttribute("href").substring(1);
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 50,
                    behavior: "smooth"
                });
            } else {
                window.location.href = this.getAttribute("href");
            }
        });
    });
});
