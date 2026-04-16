import torch
from src.dataset import IsotopeDataset
from src.model import IsotopeGNN
from src.train import train_step, mc_dropout_predict, get_real_mae


def main():
    # 1. Load Data
    dataset = IsotopeDataset(root="data")
    data = dataset[0]  # The giant graph

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = data.to(device)

    # 2. Initialize Model
    # input_dim=5: [v, J, J(J+1), ECalc, mu]
    model = IsotopeGNN(input_dim=5, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 3. Training Loop
    for epoch in range(1, 501):
        loss = train_step(model, data, optimizer)

        if epoch % 50 == 0:
            mae, _, _ = get_real_mae(model, data, "train_mask")
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train MAE: {mae:.6f} cm-1")

    # 4. Final Bayesian Inference (MC Dropout)
    print("\nRunning MC Dropout Inference for all levels...")
    final_mu_norm, total_unc_norm = mc_dropout_predict(model, data, n_passes=50)

    # Unscale everything for the final line list
    final_predictions = (final_mu_norm * data.y_std) + data.y_mean
    # Uncertainty scales by std (but doesn't shift by mean)
    final_uncertainty = total_unc_norm * data.y_std

    # final_predictions now contains E_marv - E_calc in cm-1
    # Add E_calc back to get predicted E_marv
    e_calc_raw = (data.x[:, 3] * data.x_std[3]) + data.x_mean[3]
    predicted_e_marvel = final_predictions.squeeze() + e_calc_raw

    print(
        f"Sample Prediction: {predicted_e_marvel[0].item():.4f} +/- {final_uncertainty[0].item():.4f}"
    )


if __name__ == "__main__":
    main()
