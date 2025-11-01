export interface MetricData {
  epoch: number
  train_loss: number
  val_loss: number
  train_accuracy: number
  val_accuracy: number
  learning_rate?: number
  epoch_time?: number
  samples_per_sec?: number
  progress?: number
  eta_seconds?: number
}

export interface TrainingState {
  state: 'queued' | 'running' | 'succeeded' | 'failed'
  error?: string
  test_accuracy?: number
}

export interface TrainingRequest {
  architecture: {
    input_size: number
    layers: Array<{
      type: string
      in?: number
      out?: number
    }>
  }
  hyperparams: {
    epochs: number
    batch_size: number
    optimizer: {
      type: string
      lr: number
      momentum: number
    }
    loss: string
    seed: number
    train_split: number
    shuffle: boolean
  }
}

export interface TrainingResponse {
  run_id: string
  events_url: string
}

export interface TrainingEvent {
  type: 'metric' | 'state'
  data: MetricData | TrainingState
}
