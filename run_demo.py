import traceback
try:
    import demo
    demo.main()
except Exception as e:
    traceback.print_exc()
