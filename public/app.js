const homeTeam = document.getElementById("homeTeam");
const awayTeam = document.getElementById("awayTeam");
const predictBtn = document.getElementById("predictBtn");
const result = document.getElementById("result");

async function loadTeams() {
    const res = await fetch("/api/teams");
    const data = await res.json();

    homeTeam.innerHTML = '<option value="">Select home team</option>';
    awayTeam.innerHTML = '<option value="">Select away team</option>';

    data.teams.forEach((team) => {
        const opt1 = document.createElement("option");
        opt1.value = team;
        opt1.textContent = team;
        homeTeam.appendChild(opt1);

        const opt2 = document.createElement("option");
        opt2.value = team;
        opt2.textContent = team;
        awayTeam.appendChild(opt2);
    });
}

predictBtn.addEventListener("click", async () => {
    const home = homeTeam.value;
    const away = awayTeam.value;

    if (!home || !away) {
        result.textContent = "Please select both teams.";
        return;
    }

    if (home === away) {
        result.textContent = "Home team and away team cannot be the same.";
        return;
    }

    result.textContent = "Predicting...";

    const res = await fetch("/api/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            home_team: home,
            away_team: away,
        }),
    });

    const data = await res.json();

    if (!res.ok) {
        result.textContent = data.detail || "Something went wrong.";
        return;
    }

    result.innerHTML = `
    <strong>Predicted Winner: ${data.winner}</strong><br><br>
    ${data.home_team} win probability: ${(data.home_probability * 100).toFixed(2)}%<br>
    ${data.away_team} win probability: ${(data.away_probability * 100).toFixed(2)}%
  `;
});

loadTeams();