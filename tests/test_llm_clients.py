#!/usr/bin/env python3
"""Smoke tests for CommonLib's llm_clients package."""
import os, sys
from pathlib import Path

# Allow running as 'python tests/test_llm_clients.py' from the CommonLib dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

passed = 0; failed = 0; failures = []

def check(name, ok, detail=''):
    global passed, failed
    if ok:
        print(f'PASS: {name}'); passed += 1
    else:
        print(f'FAIL: {name}{("  " + detail) if detail else ""}')
        failed += 1; failures.append(name)

# 1. Top-level package importable
try:
    import commonlib
    check('import commonlib', True)
except Exception as e:
    check('import commonlib', False, str(e))

# 2. llm_clients subpackage importable
try:
    from commonlib import llm_clients
    check('import commonlib.llm_clients', True)
except Exception as e:
    check('import commonlib.llm_clients', False, str(e))

# 3. Public API present
try:
    from commonlib.llm_clients import (
        QuotaExceededError, OpenAILLMClient, GeminiLLMClient,
        OpenRouterLLMClient, create_llm_client,
    )
    check('public API: QuotaExceededError', True)
    check('public API: OpenAILLMClient class', True)
    check('public API: GeminiLLMClient class', True)
    check('public API: OpenRouterLLMClient class', True)
    check('public API: create_llm_client fn', callable(create_llm_client))
except Exception as e:
    check('public API import', False, str(e))

# 4. Exception is the expected subclass
try:
    from commonlib.llm_clients import QuotaExceededError
    check('QuotaExceededError is RuntimeError subclass', issubclass(QuotaExceededError, RuntimeError))
except Exception as e:
    check('QuotaExceededError class shape', False, str(e))

# 5. The shim still works (for any unmigrated code)
try:
    from seo_generation.seo_audit.llm_clients import QuotaExceededError as ShimErr
    from commonlib.llm_clients import QuotaExceededError as RealErr
    check('shim re-exports same class', ShimErr is RealErr)
except Exception as e:
    check('shim still functional', False, str(e))

# 6. create_llm_client raises predictably for unknown provider
try:
    from commonlib.llm_clients import create_llm_client
    try:
        create_llm_client(provider='not-a-real-provider', model='x')
        check('create_llm_client rejects unknown provider', False, 'no exception raised')
    except Exception:
        check('create_llm_client rejects unknown provider', True)
except Exception as e:
    check('create_llm_client smoke', False, str(e))

print()
print(f'=== CommonLib: {passed}/{passed+failed} passed ({failed} failures) ===')
if failures:
    for f in failures: print(f'  - {f}')
sys.exit(0 if failed == 0 else 1)
