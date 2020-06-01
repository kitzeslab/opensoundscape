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
)

commands_accepting_dir=(
  raven_annotation_check
  raven_lowercase_annotations
  raven_generate_class_corrections
  raven_query_annotations
)

_opensoundscape_complete() {
  local cur_word prev_word

  cur_word="${COMP_WORDS[COMP_CWORD]}"
  prev_word="${COMP_WORDS[COMP_CWORD-1]}"

  if [[ ${prev_word} == "opensoundscape" ]]; then
    COMPREPLY=( $(compgen -W "$(echo ${all_commands[@]})" -- ${cur_word} ) )
  elif [[ " ${commands_accepting_dir[@]} " =~ " ${prev_word} " ]]; then
    COMPREPLY=( $(compgen -d -- ${cur_word} ) )
  fi

  return 0
}

complete -F _opensoundscape_complete opensoundscape
"""
