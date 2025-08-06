#!/bin/bash

jq -n --arg prompt "$(cat /home/sysadmin/BurnIA-script/final_prompt.txt)" \
  '{model: "llama3", prompt: $prompt, stream: false}' | \
  curl -s -H "Content-Type: application/json" -d @- http://localhost:11434/api/generate | \
  jq -r '.response' > response_ia.txt
