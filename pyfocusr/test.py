def get_all_pairwise_surface_errors(list_mesh_names, location_meshes):
    errors = np.zeros((len(list_mesh_names), len(list_mesh_names)))
    for mesh_idx_1, mesh_name_1 in enumerate(list_mesh_names):
        print(
            "Beginning Mesh: {},\t{}/{}".format(mesh_name_1, mesh_idx_1 + 1, len(list_mesh_names))
        )
        starting_idx_2 = mesh_idx_1 + 1
        for mesh_idx_2 in range(starting_idx_2, len(list_mesh_names)):
            mesh_name_2 = list_mesh_names[mesh_idx_2]
            print(
                "Begining Second Mesh: {},\t{}/{}".format(
                    mesh_name_2, mesh_idx_2, len(list_mesh_names)
                )
            )
            mesh_1 = read_vtk(os.path.join(location_meshes, mesh_name_1))
            mesh_2 = read_vtk(os.path.join(location_meshes, mesh_name_2))

            transform = icp_transform(mesh_1, mesh_2)
            mesh_1 = apply_icp_transform(mesh_1, transform)

            error_for_each_pt_on_mesh_1 = get_surface_distance_metrics(mesh_1, mesh_2)
            error_for_each_pt_on_mesh_2 = get_surface_distance_metrics(mesh_2, mesh_1)

            errors[mesh_idx_1, mesh_idx_2] = error_for_each_pt_on_mesh_1
            errors[mesh_idx_2, mesh_idx_1] = error_for_each_pt_on_mesh_2

            print(errors[mesh_idx_1, mesh_idx_2])
            print(errors[mesh_idx_2, mesh_idx_1])

    return errors
