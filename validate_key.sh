#!/bin/bash
KEY=$(cat ~/.gemini_key | tr -d '\n\r ')
echo "Testing Key: ${KEY:0:10}..."

curl -s -H 'Content-Type: application/json' \
     -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
     "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${KEY}" > curl_test.json

cat curl_test.json
