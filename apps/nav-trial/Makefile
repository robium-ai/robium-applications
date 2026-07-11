COMPOSE = docker compose -f docker/compose.yaml

.PHONY: build sim slam nav smoke check-map down

build:
	$(COMPOSE) build

sim:
	$(COMPOSE) --profile sim up --abort-on-container-exit

slam:
	$(COMPOSE) --profile slam up --abort-on-container-exit

nav:
	$(COMPOSE) --profile nav up --abort-on-container-exit

smoke:
	$(COMPOSE) --profile test up --build --abort-on-container-exit --exit-code-from smoke

check-map:
	python3 tests/check_map.py src/nav_trial_bringup/maps/map.yaml

down:
	$(COMPOSE) --profile "*" down --remove-orphans
