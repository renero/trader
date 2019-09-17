from dictionary import Dictionary

if __name__ == "__main__":
    configuration = Dictionary()
    input_data = read_data(configuration)
    konkorde = Konkorde(configuration)
    output = konkorde.compute(input_data)
    save_data(output)