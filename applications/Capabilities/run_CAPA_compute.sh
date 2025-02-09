#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status


echo "Running: python src/openllm_model_dump.py"
python src/openllm_model_dump.py

echo "Running: python src/data_extraction.py dumps/openllmlb --startswith samples_leaderboard_mmlu_pro --dataset mmlu_pro"
python src/data_extraction.py dumps/openllmlb --startswith samples_leaderboard_mmlu_pro --dataset mmlu_pro

BASE_CMD="python src/data_extraction.py dumps/openllmlb --dataset bbh  --startswith"
LEADERBOARD_NAMES=(
    "samples_leaderboard_bbh_boolean_expressions"
    "samples_leaderboard_bbh_formal_fallacies"
    "samples_leaderboard_bbh_causal_judgement"
    "samples_leaderboard_bbh_date_understanding"
    "samples_leaderboard_bbh_disambiguation_qa"
    "samples_leaderboard_bbh_geometric_shapes"
    "samples_leaderboard_bbh_hyperbaton"
    "samples_leaderboard_bbh_logical_deduction_five_objects"
    "samples_leaderboard_bbh_logical_deduction_seven_objects"
    "samples_leaderboard_bbh_logical_deduction_three_objects"
    "samples_leaderboard_bbh_movie_recommendation"
    "samples_leaderboard_bbh_navigate"
    "samples_leaderboard_bbh_penguins_in_a_table"
    "samples_leaderboard_bbh_reasoning_about_colored_objects"
    "samples_leaderboard_bbh_ruin_names"
    "samples_leaderboard_bbh_salient_translation_error_detection"
    "samples_leaderboard_bbh_snarks"
    "samples_leaderboard_bbh_sports_understanding"
    "samples_leaderboard_bbh_temporal_sequences"
    "samples_leaderboard_bbh_tracking_shuffled_objects_five_objects"
    "samples_leaderboard_bbh_tracking_shuffled_objects_seven_objects"
    "samples_leaderboard_bbh_tracking_shuffled_objects_three_objects"
    "samples_leaderboard_bbh_web_of_lies"
)

for NAME in "${LEADERBOARD_NAMES[@]}"; do
    echo "Running: $BASE_CMD $NAME"
    $BASE_CMD "$NAME"
done

echo "Running: python src/dataset_dump.py"
python src/dataset_dump.py

echo "Running: python src/combine_BBH.py --data BBH"
python src/combine_BBH.py --data BBH

echo "Running: python src/filter_MMLU.py"
python src/filter_MMLU.py cleaned_dumps/OPENLLMLB_MCQ_MMLU


echo "Running: python src/model_bin_generation.py cleaned_dumps/OPENLLMLB_MCQ_BBH_cleaned"
python src/model_bin_generation.py cleaned_dumps/OPENLLMLB_MCQ_BBH_cleaned

echo "Running: python src/model_bin_generation.py cleaned_dumps/OPENLLMLB_MCQ_MMLU_cleaned"
python src/model_bin_generation.py cleaned_dumps/OPENLLMLB_MCQ_MMLU_cleaned

echo "Running: python src/CAPA_gen.py cleaned_dumps/OPENLLMLB_MCQ_MMLU_cleaned --type prob"
python src/CAPA_gen.py cleaned_dumps/OPENLLMLB_MCQ_MMLU_cleaned --type prob

echo "Running: python src/CAPA_gen.py cleaned_dumps/OPENLLMLB_MCQ_MMLU_cleaned --type discrete"
python src/CAPA_gen.py cleaned_dumps/OPENLLMLB_MCQ_MMLU_cleaned --type discrete


echo "Running: python src/CAPA_gen.py cleaned_dumps/OPENLLMLB_MCQ_BBH_cleaned --type prob"
python src/CAPA_gen.py cleaned_dumps/OPENLLMLB_MCQ_BBH_cleaned --type prob

echo "Running: python src/CAPA_gen.py cleaned_dumps/OPENLLMLB_MCQ_BBH_cleaned --type discrete"
python src/CAPA_gen.py cleaned_dumps/OPENLLMLB_MCQ_BBH_cleaned --type discrete

