# match ...case (python version >=3.10)

day = int(input("enter day:"))

match day:

    case 1:
        print("monday")

    case 2:
        print("tuesday")

    case _:
        print("default")