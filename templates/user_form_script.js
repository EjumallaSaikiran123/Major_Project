document.addEventListener("DOMContentLoaded", function () {
    document.querySelector("form").addEventListener("submit", function (event) {
        event.preventDefault(); // Prevent default form submission

        let totalScore = 0;
        let radios = document.querySelectorAll("input[type='radio']:checked");
        
        radios.forEach(radio => {
            totalScore += parseInt(radio.value);
        });
        
        let difficulty = "Normal";
        if (totalScore > 25 && totalScore <= 50) {
            difficulty = "Medium";
        } else if (totalScore > 50 && totalScore <= 75) {
            difficulty = "Hard";
        } else if (totalScore > 75 && totalScore <= 100) {
            difficulty = "Very Hard";
        } else if (totalScore > 100 && totalScore <= 125) {
            difficulty = "Extreme";
        }
        
        // Redirect to report.html with query parameters
        window.location.href = `report.html?score=${totalScore}&difficulty=${difficulty}`;
    });
});
