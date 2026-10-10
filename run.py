"""
run.py
──────
Application runner for ERC Analyzer.
Sets UTF-8 stdout and Windows selector event loop policy before launching uvicorn.
"""
import sys
import asyncio

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        import uvicorn.loops.asyncio
        uvicorn.loops.asyncio.asyncio_loop_factory = lambda use_subprocess=False: asyncio.SelectorEventLoop
        import uvicorn.loops.auto
        uvicorn.loops.auto.auto_loop_factory = lambda use_subprocess=False: asyncio.SelectorEventLoop
    except Exception:
        pass

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("\n" + "=" * 60)
    print("Starting ERC Analyzer - Regulatory Compliance RAG Engine")
    print(f"Web Application: http://0.0.0.0:{port}")
    print(f"API Documentation: http://0.0.0.0:{port}/docs")
    print("=" * 60 + "\n")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        loop="none" if sys.platform == "win32" else "auto",
    )
