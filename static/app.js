document.addEventListener('DOMContentLoaded', () => {

    // Fetch and display scan results
    async function fetchScanResults() {
        try {
            const response = await fetch('/api/scan-results');
            const result = await response.json();

            if (result.status === 'success') {
                const data = result.data;

                // Update header metrics
                document.getElementById('market-status').textContent = data.market_status;
                document.getElementById('market-status').className = 'status-badge ' + data.market_status.toLowerCase();
                document.getElementById('vix-level').textContent = `VIX: ${data.vix_level}`;

                // Update scan stats
                document.getElementById('scan-timestamp').textContent = new Date(data.timestamp).toLocaleString();
                document.getElementById('total-scanned').textContent = data.total_scanned;
                document.getElementById('total-passed').textContent = data.passed_count;

                // Populate Table
                const tbody = document.getElementById('scan-body');
                tbody.innerHTML = ''; // clear loading

                if (data.setups.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7">No setups found today.</td></tr>';
                    return;
                }

                data.setups.forEach(setup => {
                    const tr = document.createElement('tr');

                    // Grade Badge Logic
                    let gradeClass = 'badge-grade-c';
                    if (setup.confidence_grade === 'A') gradeClass = 'badge-grade-a';
                    if (setup.confidence_grade === 'B') gradeClass = 'badge-grade-b';

                    tr.innerHTML = `
                        <td class="ticker">${setup.ticker}</td>
                        <td class="setup-type">${setup.strategy}</td>
                        <td class="price-val">$${setup.trigger.toFixed(2)}</td>
                        <td><span class="badge ${gradeClass}">${setup.confidence_grade}</span></td>
                        <td>${setup.win_rate}%</td>
                        <td>${setup.momentum_score} / ${setup.rs_percentile}</td>
                        <td class="risk-reward">
                            <span class="stop">$${setup.stop.toFixed(2)}</span>
                            <span class="target">$${setup.target.toFixed(2)}</span>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            } else {
                document.getElementById('scan-body').innerHTML = `<tr><td colspan="7">${result.message}</td></tr>`;
            }
        } catch (error) {
            console.error('Error fetching scan results:', error);
        }
    }

    // Fetch and display Alpaca portfolio data
    async function fetchPortfolio() {
        try {
            const response = await fetch('/api/portfolio');
            const result = await response.json();

            if (result.status === 'success') {
                // Update Portfolio Summary
                const acc = result.account;
                document.getElementById('portfolio-value').textContent = `$${acc.portfolio_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

                document.getElementById('bp-val').textContent = `$${parseFloat(acc.buying_power).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
                document.getElementById('bp-val').classList.remove('loading-text');

                document.getElementById('cash-val').textContent = `$${acc.cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
                document.getElementById('cash-val').classList.remove('loading-text');

                document.getElementById('dt-val').textContent = (3 - parseInt(acc.day_trade_count)).toString();
                document.getElementById('dt-val').classList.remove('loading-text');

                // Update Positions Table
                const tbody = document.getElementById('positions-body');
                tbody.innerHTML = ''; // clear loading

                if (result.positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No Active Positions</td></tr>';
                    return;
                }

                result.positions.forEach(p => {
                    const tr = document.createElement('tr');

                    // P&L Color
                    const plColor = p.unrealized_pl >= 0 ? 'profit' : 'loss';
                    const sign = p.unrealized_pl >= 0 ? '+' : '';

                    tr.innerHTML = `
                        <td class="ticker">${p.symbol}</td>
                        <td>${p.qty}</td>
                        <td>$${p.avg_entry_price.toFixed(2)}</td>
                        <td>$${p.current_price.toFixed(2)}</td>
                        <td class="${plColor}">${sign}${p.unrealized_plpc.toFixed(2)}%</td>
                        <td>$${p.market_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                    `;
                    tbody.appendChild(tr);
                });

            } else {
                document.getElementById('positions-body').innerHTML = `<tr><td colspan="6" class="error-msg">${result.message}</td></tr>`;
                document.getElementById('portfolio-value').textContent = "Connection Error";
            }

        } catch (error) {
            console.error('Error fetching portfolio:', error);
        }
    }

    // Initial fetch
    fetchScanResults();
    fetchPortfolio();

    // Refresh data every 60 seconds
    setInterval(() => {
        fetchScanResults();
        fetchPortfolio();
    }, 60000);
});
