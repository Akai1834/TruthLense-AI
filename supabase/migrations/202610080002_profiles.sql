create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  name text not null,
  email text not null,
  created_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

create policy if not exists "Profiles are viewable by owner"
  on public.profiles for select
  using (auth.uid() = id);

create policy if not exists "Profiles can be inserted by owner"
  on public.profiles for insert
  with check (auth.uid() = id);

create policy if not exists "Profiles can be updated by owner"
  on public.profiles for update
  using (auth.uid() = id)
  with check (auth.uid() = id);
