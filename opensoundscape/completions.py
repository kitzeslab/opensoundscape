#!/usr/bin/env python3

COMPLETIONS = """all_commands=(
  -h
  --help
  -v
  --version
  completions
  raven_annotation_check
  raven_lowercase_annotations
  raven_generate_class_corrections
  raven_query_annotations
  split_audio
  predict_from_directory
  split_and_save
)

split_audio_options=(
  -i
  --input_directory
  -o
  --output_directory
  -s
  --segments
  -c
  --config
)

predict_from_directory_options=(
  -i
  --input_directory
  -d
  --state_dict
  -c
  --config
)

split_and_save_options=(
  -a
  --audio_file
  -o
  --output_directory
  -s
  --segments
  -c
  --config
)

commands_accepting_dir=(
  raven_annotation_check
  raven_lowercase_annotations
  raven_generate_class_corrections
  raven_query_annotations
  -i
  --input_directory
  -o
  --output_directory
)

commands_accepting_file_or_dir=(
  -s
  --segments
  -c
  --config
  -a
  --audio_file
  -d
  --state_dict
)

_opensoundscape_complete() {
  local cur_word prev_word subcommand

  cur_word="${COMP_WORDS[COMP_CWORD]}"
  prev_word="${COMP_WORDS[COMP_CWORD-1]}"
  subcommand="${COMP_WORDS[1]}"

  if [[ ${prev_word} == "opensoundscape" ]]; then
    COMPREPLY=( $(compgen -W "$(echo ${all_commands[@]})" -- ${cur_word} ) )
  elif [[ " ${commands_accepting_dir[@]} " =~ " ${prev_word} " ]]; then
    COMPREPLY=( $(compgen -d -- ${cur_word} ) )
  elif [[ " ${commands_accepting_file_or_dir[@]} " =~ " ${prev_word} " ]]; then
    COMPREPLY=( $(compgen -f -- ${cur_word} ) )
  elif [[ ${subcommand} == "split_audio" ]]; then
    COMPREPLY=( $(compgen -W "$(echo ${split_audio_options[@]})" -- ${cur_word} ) )
  elif [[ ${subcommand} == "predict_from_directory" ]]; then
    COMPREPLY=( $(compgen -W "$(echo ${predict_from_directory_options[@]})" -- ${cur_word} ) )
  elif [[ ${subcommand} == "split_and_save" ]]; then
    COMPREPLY=( $(compgen -W "$(echo ${split_audio_options[@]})" -- ${cur_word} ) )
  fi

  return 0
}

complete -F _opensoundscape_complete opensoundscape
"""
