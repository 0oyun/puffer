// Provider setup must only offer providers that the backend can actually
// connect and run. Google and GitHub Copilot have no provider descriptor, no
// Gemini text-execution adapter, and no Copilot OAuth path, so they should not
// appear as connectable setup entries.
import { expect, test } from "@playwright/test";
import { FakeDaemon } from "./support/fakeDaemon";

test("provider setup does not offer providers without backend support", async ({ page }) => {
  const daemon = new FakeDaemon({ auth: [], providers: [] });
  await daemon.install(page);
  await daemon.open(page, { forceOnboarding: true, skipOnboarding: false });

  // Real, backed providers should still be offered.
  await expect(page.locator(".provider-card").filter({ hasText: "Anthropic" })).toHaveCount(1);
  await expect(page.locator(".provider-card").filter({ hasText: "OpenAI" }).first()).toBeVisible();

  // Unsupported providers must not be presented as setup entries.
  await expect(
    page.locator(".provider-card").filter({ hasText: "GitHub Copilot" })
  ).toHaveCount(0);
  await expect(page.locator(".provider-card").filter({ hasText: "Google" })).toHaveCount(0);
});
