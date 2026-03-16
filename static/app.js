document.addEventListener("DOMContentLoaded", () => {
    function formatMoney(value) {
        return `$${Number(value || 0).toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })}`;
    }

    function marketStatusClass(status) {
        const text = String(status || "").toUpperCase();
        if (text.includes("PANIC") || text.includes("BEAR")) return "bear";
        if (text.includes("FEAR") || text.includes("CORRECTION")) return "correction";
        if (text.includes("BULL") || text.includes("RECOVERY")) return "uptrend";
        return "neutral";
    }

    function gradeClass(grade) {
        if (grade === "A") return "badge-grade-a";
        if (grade === "B") return "badge-grade-b";
        return "badge-grade-c";
    }

    function opportunityStateClass(state) {
        const text = String(state || "").toUpperCase();
        if (text === "ACTIONABLE") return "state-actionable";
        if (text === "RESEARCH") return "state-research";
        if (text === "DEFENSIVE") return "state-defensive";
        return "state-quiet";
    }

    function fallbackOpportunity(data) {
        const actionable = Number(data.actionable_count ?? data.passed_count ?? (Array.isArray(data.setups) ? data.setups.length : 0));
        const research = Number(data.research_watchlist_count ?? data.watchlist_count ?? (Array.isArray(data.research_watchlist) ? data.research_watchlist.length : 0));
        const marketStatus = String(data.market_status || "").toUpperCase();
        const vixLevel = Number(data.vix_level || 0);

        if (actionable > 0) {
            return {
                state: "ACTIONABLE",
                headline: `${actionable} actionable setup${actionable === 1 ? "" : "s"} ready for review.`,
                detail: research > 0 ? `${research} extra research names are also available.` : "Focus on the actionable list first.",
            };
        }
        if (research > 0) {
            const researchVerb = research === 1 ? "name is" : "names are";
            return {
                state: "RESEARCH",
                headline: `No action-ready setups today, but ${research} research ${researchVerb} worth stalking.`,
                detail: "The engine stayed selective, but the watchlist still has momentum names.",
            };
        }
        if (vixLevel >= 30 || marketStatus.includes("BEAR") || marketStatus.includes("PANIC") || marketStatus.includes("FEAR")) {
            return {
                state: "DEFENSIVE",
                headline: "The market is defensive and the long book is intentionally quiet.",
                detail: "A blank long list can be the right answer in a risk-off tape.",
            };
        }
        return {
            state: "QUIET",
            headline: "No action-ready setups or research names cleared the bar.",
            detail: "This is a genuinely quiet scan, not just an empty trade list.",
        };
    }

    function renderSetups(data) {
        const tbody = document.getElementById("scan-body");
        tbody.innerHTML = "";

        if (!Array.isArray(data.setups) || data.setups.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No action-ready setups today.</td></tr>';
            return;
        }

        data.setups.forEach((setup) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td class="ticker">${setup.ticker}</td>
                <td class="setup-type">${setup.strategy}</td>
                <td class="price-val">$${Number(setup.trigger || 0).toFixed(2)}</td>
                <td><span class="badge ${gradeClass(setup.confidence_grade)}">${setup.confidence_grade}</span></td>
                <td>${Number(setup.win_rate || 0).toFixed(1)}%</td>
                <td>${Number(setup.momentum_score || 0).toFixed(1)} / ${Number(setup.rs_percentile || 0).toFixed(1)}</td>
                <td class="risk-reward">
                    <span class="stop">$${Number(setup.stop || 0).toFixed(2)}</span>
                    <span class="target">$${Number(setup.target || 0).toFixed(2)}</span>
                </td>
            `;
            tbody.appendChild(tr);
        });
    }

    function renderWatchlist(data) {
        const tbody = document.getElementById("watchlist-body");
        tbody.innerHTML = "";
        const watchlist = Array.isArray(data.research_watchlist) ? data.research_watchlist : [];

        if (watchlist.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No research watchlist names available.</td></tr>';
            return;
        }

        watchlist.forEach((item) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td class="ticker">${item.ticker}</td>
                <td class="setup-type">${item.theme}</td>
                <td><span class="badge watch-status">${item.status}</span></td>
                <td class="price-val">$${Number(item.trigger || 0).toFixed(2)}</td>
                <td>${Number(item.score || 0).toFixed(1)}</td>
                <td>${Number(item.momentum_score || 0).toFixed(1)} / ${Number(item.rs_percentile || 0).toFixed(1)}</td>
                <td class="watchlist-why">${item.why || ""}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    async function fetchScanResults() {
        try {
            const response = await fetch("/api/scan-results");
            const result = await response.json();

            if (result.status !== "success") {
                document.getElementById("scan-body").innerHTML = `<tr><td colspan="7">${result.message}</td></tr>`;
                document.getElementById("watchlist-body").innerHTML = '<tr><td colspan="7" class="empty-state">Awaiting research watchlist...</td></tr>';
                return;
            }

            const data = result.data;
            const marketStatus = data.market_status || "UNKNOWN";
            const statusNode = document.getElementById("market-status");
            statusNode.textContent = marketStatus;
            statusNode.className = `status-badge ${marketStatusClass(marketStatus)}`;

            document.getElementById("vix-level").textContent = `VIX: ${data.vix_level ?? "--"}`;
            document.getElementById("scan-timestamp").textContent = new Date(data.timestamp).toLocaleString();
            document.getElementById("total-scanned").textContent = data.total_scanned ?? 0;
            document.getElementById("total-passed").textContent = data.actionable_count ?? data.passed_count ?? 0;
            document.getElementById("research-count").textContent = data.research_watchlist_count ?? data.watchlist_count ?? 0;

            const fallback = fallbackOpportunity(data);
            const opportunityState = data.opportunity_state || data.opportunity?.state || fallback.state;
            const opportunityHeadline = data.opportunity_headline || data.opportunity?.headline || fallback.headline;
            const opportunityDetail = data.opportunity_detail || data.opportunity?.detail || fallback.detail;
            const opportunityStateNode = document.getElementById("opportunity-state");
            opportunityStateNode.textContent = opportunityState;
            opportunityStateNode.className = opportunityStateClass(opportunityState);
            document.getElementById("opportunity-headline").textContent = opportunityHeadline;
            document.getElementById("opportunity-detail").textContent = opportunityDetail;

            renderSetups(data);
            renderWatchlist(data);
        } catch (error) {
            console.error("Error fetching scan results:", error);
        }
    }

    async function fetchPortfolio() {
        try {
            const response = await fetch("/api/portfolio");
            const result = await response.json();

            if (result.status !== "success") {
                document.getElementById("positions-body").innerHTML = `<tr><td colspan="6" class="error-msg">${result.message}</td></tr>`;
                document.getElementById("portfolio-value").textContent = "Connection Error";
                return;
            }

            const acc = result.account;
            document.getElementById("portfolio-value").textContent = formatMoney(acc.portfolio_value);
            document.getElementById("bp-val").textContent = formatMoney(acc.buying_power);
            document.getElementById("cash-val").textContent = formatMoney(acc.cash);
            document.getElementById("dt-val").textContent = Math.max(0, 3 - Number(acc.day_trade_count || 0)).toString();
            document.getElementById("broker-mode").textContent = `Alpaca ${String(acc.account_mode || "paper").toUpperCase()}`;

            const tbody = document.getElementById("positions-body");
            tbody.innerHTML = "";

            if (!Array.isArray(result.positions) || result.positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No Active Positions</td></tr>';
                return;
            }

            result.positions.forEach((position) => {
                const tr = document.createElement("tr");
                const plColor = Number(position.unrealized_pl || 0) >= 0 ? "profit" : "loss";
                const sign = Number(position.unrealized_pl || 0) >= 0 ? "+" : "";
                tr.innerHTML = `
                    <td class="ticker">${position.symbol}</td>
                    <td>${Number(position.qty || 0)}</td>
                    <td>$${Number(position.avg_entry_price || 0).toFixed(2)}</td>
                    <td>$${Number(position.current_price || 0).toFixed(2)}</td>
                    <td class="${plColor}">${sign}${Number(position.unrealized_plpc || 0).toFixed(2)}%</td>
                    <td>${formatMoney(position.market_value)}</td>
                `;
                tbody.appendChild(tr);
            });
        } catch (error) {
            console.error("Error fetching portfolio:", error);
        }
    }

    fetchScanResults();
    fetchPortfolio();

    setInterval(() => {
        fetchScanResults();
        fetchPortfolio();
    }, 60000);
});
