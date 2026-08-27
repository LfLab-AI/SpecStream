BUILD_DIR := $(CURDIR)/build
CUDA_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | sed 's/\.//')

.PHONY: config build validate validate-global clean rebuild

config:
	cmake -S . -B $(BUILD_DIR) -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
		-DSPECSTREAM_SMCTRL_BUILD_TESTS=ON

build:
	cmake --build $(BUILD_DIR) -j

validate:
	@test -n "$(TPC_HIGH)" || (echo "usage: make validate TPC_HIGH=<count> [TPC_LOW=0]"; exit 2)
	$(BUILD_DIR)/specstream_smctrl_validator $(or $(TPC_LOW),0) $(TPC_HIGH)

validate-global:
	@test -n "$(TPC_HIGH)" || (echo "usage: make validate-global TPC_HIGH=<count> [TPC_LOW=0]"; exit 2)
	$(BUILD_DIR)/specstream_smctrl_validator $(or $(TPC_LOW),0) $(TPC_HIGH) global

clean:
	cmake --build $(BUILD_DIR) --target clean

rebuild: clean build
