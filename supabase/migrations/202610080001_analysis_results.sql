create table if not exists public.analysis_results (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  media_type text not null check (media_type in ('image', 'video')),
  score_label text not null check (score_label in ('ai_generated', 'manipulated')),
  positive_percent numeric(5, 1) not null check (positive_percent between 0 and 100),
  real_percent numeric(5, 1) not null check (real_percent between 0 and 100),
  frames_analyzed integer not null check (frames_analyzed > 0),
  model_id text not null,
  created_at timestamptz not null default now(),
  constraint analysis_results_media_label_check check (
    (media_type = 'image' and score_label = 'ai_generated')
    or (media_type = 'video' and score_label = 'manipulated')
  ),
  constraint analysis_results_scores_sum_check check (
    abs(positive_percent + real_percent - 100) <= 0.1
  )
);

create index if not exists analysis_results_user_created_at_idx
  on public.analysis_results (user_id, created_at desc);

alter table public.analysis_results enable row level security;

revoke all on table public.analysis_results from anon;
grant select, insert on table public.analysis_results to authenticated;

drop policy if exists "Users can read their own analysis results" on public.analysis_results;
create policy "Users can read their own analysis results"
  on public.analysis_results
  for select
  to authenticated
  using ((select auth.uid()) = user_id);

drop policy if exists "Users can create their own analysis results" on public.analysis_results;
create policy "Users can create their own analysis results"
  on public.analysis_results
  for insert
  to authenticated
  with check ((select auth.uid()) = user_id);
