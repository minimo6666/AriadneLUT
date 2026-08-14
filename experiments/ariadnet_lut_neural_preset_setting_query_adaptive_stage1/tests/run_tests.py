from test_query_adaptive import (
    test_query_mass_normalizes_and_is_sparse_for_constant_color,
    test_generator_shapes_and_gradients_without_evidence,
    test_generator_with_style_evidence,
)


def main():
    tests = [
        test_query_mass_normalizes_and_is_sparse_for_constant_color,
        test_generator_shapes_and_gradients_without_evidence,
        test_generator_with_style_evidence,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
