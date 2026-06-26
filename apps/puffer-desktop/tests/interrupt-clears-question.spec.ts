// Interrupting a turn while the agent is waiting on AskUserQuestion must clear
// the pending question. Otherwise the prompt stays on screen and the session is
// stuck "awaiting" an answer that can no longer be given.
import { expect, type Page, test } from "@playwright/test";
import { FakeDaemon } from "./support/fakeDaemon";

async function openSession(page: Page, name: RegExp): Promise<void> {
  await page.getByRole("button", { name }).first().click();
}

test("interrupting a turn clears a pending AskUserQuestion prompt", async ({ page }) => {
  const daemon = new FakeDaemon();
  await daemon.install(page);
  await daemon.open(page);

  await openSession(page, /^Browser regression\b/);

  // Start a turn.
  await page.locator(".pf-composer textarea").fill("Do the thing");
  await page.getByRole("button", { name: "Send" }).click();
  await daemon.waitForRequest(
    "run_agent_turn",
    (request) => request.params.message === "Do the thing"
  );
  await expect(page.getByRole("button", { name: "Stop turn" })).toBeVisible();

  // The agent calls AskUserQuestion mid-turn.
  daemon.emit("session:session-browser:event", {
    type: "user-question-request",
    turnId: "turn-session-browser",
    requestId: "q-1",
    questions: [
      {
        header: "Confirm",
        question: "Should I proceed with the deploy?",
        type: "choice",
        options: [
          { label: "Yes", description: "Proceed" },
          { label: "No", description: "Stop" }
        ]
      }
    ]
  });

  // The question prompt is shown -> session is "awaiting" an answer.
  await expect(page.locator(".pf-question")).toBeVisible();

  // User interrupts the turn.
  await page.getByRole("button", { name: "Stop turn" }).click();
  await daemon.waitForRequest(
    "cancel_turn",
    (request) => request.params.turnId === "turn-session-browser"
  );

  // Interrupting clears the pending question so the session is usable again.
  await expect(page.locator(".pf-question")).toHaveCount(0, { timeout: 3000 });
});
