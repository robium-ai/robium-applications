COMPOSE = docker compose -f docker/compose.yaml

.PHONY: build sim slam nav demo demo-smoke demo-image demo-deploy smoke check-map down

build:
	$(COMPOSE) build sim

sim:
	$(COMPOSE) --profile sim up --abort-on-container-exit

slam:
	$(COMPOSE) --profile slam up --abort-on-container-exit

nav:
	$(COMPOSE) --profile nav up --abort-on-container-exit

demo:
	$(COMPOSE) --profile demo up --abort-on-container-exit

# Demo gate: boot the demo scenario, assert the Foxglove WebSocket answers,
# the stack reaches DEMO READY, and one nav goal succeeds. Exit code gated.
# (Bounded shell loops, not timeout(1) — absent on macOS hosts.)
demo-smoke:
	$(COMPOSE) --profile demo up --build -d demo
	n=0; until bash tests/check_ws.sh http://localhost:8765; do \
	  n=$$((n+1)); [ $$n -ge 20 ] && echo "WS TIMEOUT" && exit 1; sleep 3; done
	n=0; until $(COMPOSE) logs demo | grep -q "DEMO READY"; do \
	  n=$$((n+1)); [ $$n -ge 48 ] && echo "READY TIMEOUT" && exit 1; sleep 5; done
	$(COMPOSE) logs demo | grep "DEMO READY"
	$(COMPOSE) exec demo /entrypoint.sh ros2 run nav_trial_bringup send_goals --goals "0.3,0.5" --timeout 120
	$(COMPOSE) --profile "*" down --remove-orphans
	@echo "DEMO SMOKE PASS"

smoke:
	$(COMPOSE) --profile test up --build --abort-on-container-exit --exit-code-from smoke

# --- live demo on Cloud Run (per-visitor instances; see robium.org spec) ---
DEMO_IMAGE = us-central1-docker.pkg.dev/robium-prod/robium/demo-nav-trial:latest

demo-image:
	gcloud builds submit --project=robium-prod --config=cloudbuild.yaml .

demo-deploy:
	gcloud run deploy demo-nav-trial --image=$(DEMO_IMAGE) \
	  --region=us-central1 --project=robium-prod --platform=managed \
	  --port=8765 --concurrency=1 --min-instances=0 --max-instances=5 \
	  --timeout=1800 --cpu=4 --memory=4Gi --cpu-boost \
	  --execution-environment=gen2 \
	  --set-env-vars=GZ_RELAY=127.0.0.1,GZ_IP=127.0.0.1 \
	  --command=/entrypoint.sh --args="ros2,launch,nav_trial_bringup,demo.launch.py" \
	  --allow-unauthenticated --quiet
# GZ_RELAY/GZ_IP: gz-transport discovery is UDP-multicast-based and Cloud Run
# has no multicast — without the unicast loopback relay, gz server is
# undiscoverable and spawn/bridge wait forever (verified 2026-07-12).

check-map:
	python3 tests/check_map.py src/nav_trial_bringup/maps/map.yaml

down:
	$(COMPOSE) --profile "*" down --remove-orphans
