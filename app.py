from biomni.agent import A1


def main() -> None:
    agent = A1(path="./data", expected_data_lake_files=[])
    agent.launch_gradio_demo(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
