const homeSelect = document.getElementById('homeTeam');
const awaySelect = document.getElementById('awayTeam');
const predictBtn = document.getElementById('predictBtn');
const statusText = document.getElementById('status');
const resultCard = document.getElementById('resultCard');
const winnerText = document.getElementById('winnerText');
const homeLabel = document.getElementById('homeLabel');
const awayLabel = document.getElementById('awayLabel');
const homeProb = document.getElementById('homeProb');
const awayProb = document.getElementById('awayProb');
const dataSource = document.getElementById('dataSource');

function createDefaultOption(label) {
  const option = document.createElement('option');
  option.value = '';
  option.textContent = label;
  return option;
}

function fillTeamSelect(selectEl, teams, label) {
  selectEl.innerHTML = '';
  selectEl.appendChild(createDefaultOption(label));
  teams.forEach((team) => {
    const option = document.createElement('option');
    option.value = team;
    option.textContent = team;
    selectEl.appendChild(option);
  });
}

async function loadTeams() {
  try {
    const response = await fetch('/api/teams');
    const data = await response.json();

    fillTeamSelect(homeSelect, data.teams, 'Select home team');
    fillTeamSelect(awaySelect, data.teams, 'Select away team');
    statusText.textContent = 'Teams loaded. Choose a matchup.';
  } catch (error) {
    statusText.textContent = 'Could not load teams.';
  }
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

predictBtn.addEventListener('click', async () => {
  const home_team = homeSelect.value;
  const away_team = awaySelect.value;

  resultCard.classList.add('hidden');

  if (!home_team || !away_team) {
    statusText.textContent = 'Please select both teams.';
    return;
  }

  statusText.textContent = 'Running prediction...';

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ home_team, away_team }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Prediction failed.');
    }

    winnerText.textContent = data.predicted_winner;
    homeLabel.textContent = `${data.home_team} win probability`;
    awayLabel.textContent = `${data.away_team} win probability`;
    homeProb.textContent = formatPercent(data.home_win_probability);
    awayProb.textContent = formatPercent(data.away_win_probability);
    dataSource.textContent = `Data source: ${data.data_source}`;

    resultCard.classList.remove('hidden');
    statusText.textContent = 'Prediction complete.';
  } catch (error) {
    statusText.textContent = error.message;
  }
});

loadTeams();
