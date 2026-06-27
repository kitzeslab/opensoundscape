for notebook in docs/tutorials/*.ipynb; do
  echo "::group::Running $notebook"
  if poetry run jupyter nbconvert \
      --to notebook \
      --execute \
      --ExecutePreprocessor.timeout=1800 \
      --ExecutePreprocessor.kernel_name=python3 \
      --output /tmp/$(basename "$notebook") \
      "$notebook"; then
    passed+=("$notebook")
    echo "PASSED: $notebook"
  else
    failed+=("$notebook")
    echo "FAILED: $notebook"
  fi
  echo "::endgroup::"
done