MANUSCRIPT_DIR := manuscript
BUILD_DIR := $(MANUSCRIPT_DIR)/build

.PHONY: all clean

all: $(BUILD_DIR)/main.pdf

$(BUILD_DIR)/main.pdf: $(MANUSCRIPT_DIR)/main.tex $(MANUSCRIPT_DIR)/references.bib
	mkdir -p $(BUILD_DIR)
	cd $(MANUSCRIPT_DIR) && latexmk -pdf -outdir=build main.tex

clean:
	rm -rf $(BUILD_DIR)
