for V in S0 S1 S3 S4 S5; do

  VARIANT="$V" \
  INPUT_LENS="8192 16384 24576 30000" \
  CONCURRENCIES="1 4 8 16 32" \
  CASE_PREFIX="${V}_capacity" \
  OUTPUT_LEN=64 \
  NUM_PROMPTS=200 \
  REQUEST_RATE=inf \
  CONTINUE_ON_ERROR=1 \
  bash scripts/specstream/paper_eval/run_i1_grid.sh

done
