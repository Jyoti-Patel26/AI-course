while True:
    print("\n--- MAIN MENU ---")
    print("1. Strawberry Chocolate")
    print("2. Sweetcorn Chat")
    print("3. Pizza")
    print("4. Garlic Bread")
    print("5. Panini")
    print("6. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        print("Strawberry Chocolate")

    elif choice == "2":
        print("Sweetcorn Chart")

    elif choice == "3":
        while True:
            print("\n--- PIZZA MENU ---")
            print("1. Sweetcorn Pizza")
            print("2. Onion Pizza")
            print("3. Normal")

            p = input("Enter your choice: ")

            if p == "1":
                print("Sweetcorn Pizza")
            elif p == "2":
                print("Onion Pizza")
            elif p == "3":
                print("Normal pizza")
            else:
                print("not available")

    elif choice == "4":
        while True:
            print("\n--- GARLIC BREAD MENU ---")
            print("1. Simple Garlic Bread")
            print("2. Cheese Garlic Bread")
            print("3. only bread")

            g = input("Enter your choice: ")

            if g == "1":
                print("You selected Simple Garlic Bread")
            elif g == "2":
                print("You selected Cheese Garlic Bread")
            elif g == "3":
                print("only bread")
            else:
                print("not available")

    elif choice == "5":
        while True:
            print("\n--- PANINI MENU ---")
            print("1. Cheese Extra Panini")
            print("2. Normal Panini")
            print("3. salad panini")

            pa = input("Enter your choice: ")

            if pa == "1":
                print("You selected Cheese Extra Panini")
            elif pa == "2":
                print("You selected Normal Panini")
            elif pa == "3":
                print("salad panini")
            else:
                print("Invalid choice")

    elif choice == "6":
        print("Thank you! Visit again")
        break

    else:
        print("Invalid choice, try again")
