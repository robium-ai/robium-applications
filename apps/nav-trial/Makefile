COMPOSE = docker compose -f docker/compose.yaml

.PHONY: build sim slam nav demo demo-smoke smoke check-map down

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

check-map:
	python3 tests/check_map.py src/nav_trial_bringup/maps/map.yaml

down:
	$(COMPOSE) --profile "*" down --remove-orphans
