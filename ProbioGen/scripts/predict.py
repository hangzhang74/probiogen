from probiogen.cli import main

if __name__ == "__main__":
    main(["predict", *(__import__("sys").argv[1:])])
