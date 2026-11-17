/**
 * Playwright smoke test: two page loads with the same shareable-roll URL
 * must produce identical readAllDiceValues() output after settlement.
 *
 * Prereq: npx vite build && npm run preview
 */
import { launchPage, waitForRollFinished } from './helpers/browser.js';
import { BASE } from './helpers/server.js';

// layout-seed/density/theme pin the clutter colliders the way a real share
// link does (buildShareableRollUrl), so the two loads simulate the same world.
const REPLAY_QUERY =
    '?webgl&no-post&fair-dice&test&seed=42424242&dice=d20:1,d6:1&v=1&layout-seed=1&density=med&theme=default';
const LOAD_TIMEOUT_MS = 120000;
const SETTLE_TIMEOUT_MS = 180000;

async function waitForReplaySettled(page) {
    await page.waitForFunction(() => window.__app?.ready === true, null, {
        timeout: LOAD_TIMEOUT_MS,
    });
    const wasmReady = await page.evaluate(() => window.__app?.isWasmAvailable?.() === true);
    if (!wasmReady) {
        return { skipped: true, reason: 'WASM physics not available (run npm run build:wasm)' };
    }
    // Wait for the replayed roll itself to finish: `ready` fires before the
    // replay spawns its dice, so areDiceSettled() alone can pass on an empty
    // table. A settle timeout fails here with the engine's sleep diagnostics.
    await waitForRollFinished(page, { timeout: SETTLE_TIMEOUT_MS });
    return { skipped: false };
}

function isBenignBrowserError(message) {
    return message.includes('404 (Not Found)') || message.includes('dice_physics.js');
}

async function captureReplayValues(page) {
    return page.evaluate(() => {
        const read = window.__app?.readAllDiceValues;
        return read().map((die) => ({ type: die.type, value: die.value }));
    });
}

// One cold load in its own browser. Each Playwright launch gets a fresh
// profile, so run B is a second *cold* visit, like a friend opening the link.
// Re-navigating the same page is deliberately avoided: a warm-cache second
// visit on SwiftShader blocks the main thread for 40-140 s before `load` (or
// even `commit`) fires and burns the whole timeout — tracked in #365.
async function loadAndCapture(url, label) {
    const { browser, page, errors } = await launchPage();
    page.setDefaultTimeout(LOAD_TIMEOUT_MS);
    try {
        await page.goto(url, { waitUntil: 'load', timeout: LOAD_TIMEOUT_MS });
        const settle = await waitForReplaySettled(page);
        if (settle.skipped) {
            console.error(`FAIL: ${settle.reason}`);
            process.exit(1);
        }
        const values = await captureReplayValues(page);
        console.log(`Run ${label}:`, JSON.stringify(values));
        return { values, errors };
    } finally {
        await browser.close();
    }
}

(async () => {
    const url = `${BASE}/${REPLAY_QUERY}`;

    try {
        console.log(`Opening replay URL: ${url}`);

        const { values: runA, errors: errorsA } = await loadAndCapture(url, 'A');
        const { values: runB, errors: errorsB } = await loadAndCapture(url, 'B');

        const fatalErrors = [...errorsA, ...errorsB].filter(
            (message) => !isBenignBrowserError(message)
        );
        if (fatalErrors.length) {
            console.error('Browser errors:', fatalErrors);
            process.exit(1);
        }

        if (!runA.length || !runB.length) {
            console.error('FAILURE: expected dice values on the table');
            process.exit(1);
        }

        const identical = JSON.stringify(runA) === JSON.stringify(runB);
        if (!identical) {
            console.error('FAILURE: replay values differ between runs');
            process.exit(1);
        }

        console.log('PASS: identical replay results across two page loads');
        process.exit(0);
    } catch (err) {
        console.error('FAILURE:', err.message);
        process.exit(1);
    }
})();
